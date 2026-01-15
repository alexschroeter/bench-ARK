"""
Training benchmark for StarDist using the bench-ARK framework with TensorFlow.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
from .tensorflow_base import TensorFlowBenchmark

logger = logging.getLogger(__name__)


class ModelManager:
    """Model manager for StarDist model loading and device assignment."""

    def __init__(self):
        self.model = None
        self.current_device = None
        self.config = None

    def load_model(self, device_context: str, n_channel: int = 1,
                   n_rays: int = 32, grid: tuple = (2, 2)) -> None:
        """Load StarDist model for the specified device."""
        logger.debug(f"Loading StarDist model for device: {device_context}")

        try:
            from stardist.models import Config2D, StarDist2D
            from stardist import gputools_available
            import tensorflow as tf

            # Determine GPU settings based on device context
            use_gpu = device_context != 'cpu'

            # NOTE: We don't use tf.config.set_visible_devices() here because:
            # 1. It can only be called before TensorFlow runtime initializes
            # 2. Once set, it cannot be changed for the process
            # Instead, we use tf.device() context in _run_single_benchmark
            
            if use_gpu:
                logger.debug(f"  Configuring for GPU: {device_context}")
                if ':' in device_context:
                    gpu_id = int(device_context.split(':')[1])
                else:
                    gpu_id = 0

                gpus = tf.config.list_physical_devices('GPU')
                if gpus and gpu_id < len(gpus):
                    logger.debug(f"  GPU {gpu_id} available: {gpus[gpu_id]}")
                    try:
                        tf.config.experimental.set_memory_growth(
                            gpus[gpu_id], True)
                    except RuntimeError:
                        pass  # Already set
                else:
                    logger.warning(f"  GPU {gpu_id} not found! GPUs: {gpus}")
            else:
                logger.debug("  Configuring for CPU")

            # Create StarDist configuration
            # use_gpu controls gputools (OpenCL), not TensorFlow GPU
            self.config = Config2D(
                n_rays=n_rays,
                grid=grid,
                use_gpu=use_gpu and gputools_available(),
                n_channel_in=n_channel,
            )

            self.current_device = device_context
            logger.debug(f"  Model configuration created for {device_context}")

        except ImportError as e:
            raise ImportError(f"StarDist not available: {e}")

    def is_loaded_for_device(self, device_context: str) -> bool:
        """Check if model is already loaded for this device."""
        return self.config is not None and self.current_device == device_context

    def create_model(self, name: str, basedir: str):
        """Create a new StarDist model instance."""
        from stardist.models import StarDist2D
        return StarDist2D(self.config, name=name, basedir=basedir)


class BenchmarkData:
    """Container for training and test data."""
    def __init__(self):
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None


class StardistTraining(TensorFlowBenchmark):
    """
    StarDist training benchmark for testing the bench-ARK framework.

    This benchmark trains StarDist models on the DSB2018 dataset and
    measures training and inference performance across different devices.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        logger.debug("StardistTraining initialization starting...")
        super().__init__(name, config)
        logger.debug("StardistTraining initialization completed.")
        self.model_manager = ModelManager()
        self.benchmark_data = BenchmarkData()
        self.benchmark_params = self.config.get('benchmarks', {}).get(
            'parameters', {}).get(self.name, {})

    def _run_benchmark(self) -> Dict[str, Any]:
        """
        Run StarDist training benchmark.

        1. Download and prepare DSB2018 training data
        2. For each device:
           - Run warmup iterations
           - Run benchmark iterations (training + inference)
           - Collect times and masks
        3. Return structured results
        """
        logger.info("Starting StarDist Training Benchmark")

        # Download and prepare training data
        (self.benchmark_data.train_data,
         self.benchmark_data.train_labels,
         self.benchmark_data.test_data,
         self.benchmark_data.test_labels) = self._download_and_prepare_training_data()

        # Get benchmark parameters from config
        warmup_iterations = self.benchmark_params.get('warmup_iterations', 1)
        num_iterations = self.benchmark_params.get('num_iterations', 1)
        n_epochs = int(self.benchmark_params.get('n_epochs', 1))
        steps_per_epoch = int(self.benchmark_params.get('steps_per_epoch', 10))
        n_rays = int(self.benchmark_params.get('n_rays', 32))

        import stardist
        logger.debug(f"StarDist path: {stardist.__file__}")
        logger.debug("Benchmark parameters:")
        logger.debug(f"  Warmup iterations: {warmup_iterations}")
        logger.debug(f"  Benchmark iterations: {num_iterations}")
        logger.debug(f"  n_epochs: {n_epochs}")
        logger.debug(f"  steps_per_epoch: {steps_per_epoch}")
        logger.debug(f"  n_rays: {n_rays}")

        # Run benchmarks for each device
        all_results = {}
        devices = self.device_manager._list_available_devices()

        for device in devices:
            logger.info(f"\n--- Testing Device: {device.name} ({device.type}) ---")

            device_results = {}

            # Setup model for this device
            device_context = self._setup_device_context(device)

            # Determine number of channels from training data
            n_channel = (1 if self.benchmark_data.train_data[0].ndim == 2
                        else self.benchmark_data.train_data[0].shape[-1])

            self.model_manager = self._load_model_for_device(
                self.model_manager, device_context, n_channel, n_rays)

            # Run warmup iterations
            logger.debug(f"    Running {warmup_iterations} warmup iterations...")
            for i in range(warmup_iterations):
                try:
                    _, _, _ = self._run_single_benchmark(device_context)
                except Exception as e:
                    logger.warning(f"Warmup iteration {i+1} failed: {e}")

            # Create directory for storing models
            dataset = self.config.get('dataset', 'default_dataset')
            device_flavour = getattr(device, 'arkitekt_flavour', 'unknown')
            models_dir = (Path.cwd() / dataset / self.name /
                          f"{self.timestamp}_{device_flavour}" / "models")
            models_dir.mkdir(parents=True, exist_ok=True)

            # Create device-specific prefix for model names
            device_safe_name = "".join(c for c in device.name
                                      if c.isalnum() or c in (' ', '-', '_'))
            device_safe_name = device_safe_name.rstrip().replace(' ', '_')

            # Run benchmark iterations and collect times, masks, and models
            benchmark_times = []
            masks_collected = []
            model_paths = []
            successful_iterations = 0
            failed_iterations = 0

            logger.debug(f"      Running {num_iterations} benchmark iterations...")
            for iteration in range(num_iterations):
                try:
                    benchmark_time, mask, model_path = (
                        self._run_single_benchmark(device_context))
                    benchmark_times.append(benchmark_time)
                    masks_collected.append(mask)

                    # Copy model to results folder with device name and
                    # iteration number
                    model_name = (f"{device_safe_name}_"
                                  f"stardist_training_model_{iteration + 1}")
                    stored_model_path = models_dir / model_name

                    import shutil
                    if Path(model_path).is_dir():
                        if stored_model_path.exists():
                            shutil.rmtree(stored_model_path)
                        shutil.copytree(model_path, stored_model_path)
                    else:
                        shutil.copy2(model_path, stored_model_path)
                    model_paths.append(str(stored_model_path))

                    successful_iterations += 1
                except Exception as e:
                    logger.warning(f"Benchmark iteration {iteration+1} "
                                   f"failed: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    failed_iterations += 1

            # Check if all models are identical
            models_identical = False
            if len(model_paths) > 1:
                models_identical = self._check_models_identical(model_paths)
                models_status = ('All models identical' if models_identical
                                 else 'Models differ')
                logger.info(f"      Model identity check: {models_status}")

            # Store all masks from the best iteration (from fastest inference)
            mask_filenames = []
            if masks_collected and benchmark_times:
                # Use masks from fastest inference (2nd element of timing pair)
                inference_times = [infer_time
                                   for _, infer_time in benchmark_times]
                best_mask_idx = np.argmin(inference_times)
                best_masks = masks_collected[best_mask_idx]

                # best_masks is a list of masks (one per test image)
                # Save all masks with test image index
                if isinstance(best_masks, list) and len(best_masks) > 0:
                    for test_idx, mask in enumerate(best_masks):
                        mask_filename = self._save_mask(mask, device, test_idx)
                        if mask_filename:
                            mask_filenames.append(mask_filename)
                else:
                    # Single mask case (backward compatibility)
                    mask_filename = self._save_mask(best_masks, device, 0)
                    if mask_filename:
                        mask_filenames.append(mask_filename)

            # Store results for this configuration
            test_key = "stardist_training "
            device_model = (getattr(device, 'cpu_model', device.name)
                            if device.type.lower() == 'cpu'
                            else device.name)
            device_results[test_key] = {
                'device_name': device.name,
                'device_type': device.type,
                'device_id': device.id,
                'device_model': device_model,
                'arkitekt_flavour': getattr(device, 'arkitekt_flavour',
                                            'unknown'),
                'warmup_iterations': warmup_iterations,
                'benchmark_iterations': num_iterations,
                'successful_iterations': successful_iterations,
                'failed_iterations': failed_iterations,
                'raw_benchmark_times': benchmark_times,
                'processed_metrics': (
                    self._calculate_processed_metrics(benchmark_times)),
                'model_paths': model_paths,
                'models_identical': models_identical,
                'mask_filenames': mask_filenames,
                'success': len(benchmark_times) > 0
            }

            if benchmark_times:
                # Calculate average training and inference times separately
                training_times = [train_time
                                  for train_time, _ in benchmark_times]
                inference_times = [inference_time
                                   for _, inference_time in benchmark_times]
                avg_training_time = sum(training_times) / len(training_times)
                avg_inference_time = (sum(inference_times) /
                                      len(inference_times))
                logger.info(f"      Completed: "
                            f"avg_train={avg_training_time:.4f}s, "
                            f"avg_inference={avg_inference_time:.4f}s, "
                            f"runs={len(benchmark_times)}/{num_iterations}")
            else:
                logger.warning("      All iterations failed")

            all_results[f"{device.name}_{device.type}"] = device_results

        # Calculate overall benchmark statistics
        overall_stats = self._calculate_overall_statistics(all_results)

        total_configs = sum(len(device_results)
                            for device_results in all_results.values())
        final_results = {
            'device_results': all_results,
            'overall_statistics': overall_stats,
            'benchmark_metadata': {
                'total_devices_tested': len(devices),
                'total_configurations': total_configs,
                'benchmark_parameters': str(self.benchmark_params)
            }
        }

        logger.info("\nBenchmark completed successfully")
        logger.info(f"  Devices tested: {len(devices)}")
        total_config = (
            final_results['benchmark_metadata']['total_configurations'])
        logger.info(f"  Total configurations: {total_config}")

        return final_results

    def _download_and_prepare_training_data(self):
        """
        Download and prepare DSB2018 training data for StarDist.

        Returns:
            Tuple of (train_data, train_labels, test_data, test_labels)
        """
        from glob import glob
        from tifffile import imread
        from csbdeep.utils import Path, download_and_extract_zip_file
        from stardist import fill_label_holes

        # Define data directory for this benchmark
        dataset = self.config.get('dataset', 'default_dataset')
        data_dir = Path.cwd() / dataset / self.name / "data"

        # Define paths for training data
        dsb_dir = data_dir / "dsb2018"
        train_dir = dsb_dir / "train"
        test_dir = dsb_dir / "test"

        # Check if data already exists
        if train_dir.exists() and test_dir.exists():
            logger.info("Training data exists, loading from data folder...")
        else:
            logger.info("Training data not found, downloading...")

            # Create data directory if it doesn't exist
            data_dir.mkdir(parents=True, exist_ok=True)

            # Download the DSB2018 training data
            download_and_extract_zip_file(
                url='https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip',
                targetdir=str(data_dir),
                verbose=1,
            )
            logger.info("Training data downloaded and extracted successfully")

        # Load training images and masks
        X_train_files = sorted(glob(str(train_dir / "images" / "*.tif")))
        Y_train_files = sorted(glob(str(train_dir / "masks" / "*.tif")))

        # Load test images and masks
        X_test_files = sorted(glob(str(test_dir / "images" / "*.tif")))
        Y_test_files = sorted(glob(str(test_dir / "masks" / "*.tif")))

        # Verify file matching
        assert all(Path(x).name == Path(y).name
                   for x, y in zip(X_train_files, Y_train_files)), \
            "Training image/mask filename mismatch"
        assert all(Path(x).name == Path(y).name
                   for x, y in zip(X_test_files, Y_test_files)), \
            "Test image/mask filename mismatch"

        # Load images
        train_data = list(map(imread, X_train_files))
        train_labels = list(map(imread, Y_train_files))
        test_data = list(map(imread, X_test_files))
        test_labels = list(map(imread, Y_test_files))

        # Fill label holes (required for StarDist)
        train_labels = [fill_label_holes(y) for y in train_labels]
        test_labels = [fill_label_holes(y) for y in test_labels]

        logger.debug(f"Loaded {len(train_data)} training images and "
                     f"{len(test_data)} test images.")

        return train_data, train_labels, test_data, test_labels

    def _setup_device_context(self, device) -> str:
        """Setup device context string for TensorFlow."""
        if device.type.lower() == 'cpu':
            return 'cpu'
        elif device.type.lower() == 'cuda':
            return f'cuda:{device.id}'
        elif device.type.lower() == 'xpu':
            return f'xpu:{device.id}'
        elif device.type.lower() in ['rocm', 'hip']:
            return f'cuda:{device.id}'  # AMD GPUs
        else:
            return 'cpu'  # Fallback

    def _load_model_for_device(self, model_manager, device_context: str,
                                n_channel: int = 1, n_rays: int = 32):
        """Load StarDist model configuration for specific device."""
        logger.debug(f"Setting up model for device: {device_context}")

        # Check if model is already loaded for this device
        if not model_manager.is_loaded_for_device(device_context):
            model_manager.load_model(device_context, n_channel=n_channel,
                                     n_rays=n_rays)
        else:
            logger.debug(f"Model already loaded for device: {device_context}")

        return model_manager

    def _run_single_benchmark(self, device_context: str) -> tuple:
        """
        Run a single training benchmark and return the execution times, mask,
        and model path.

        Args:
            device_context: Device context string

        Returns:
            Tuple of (benchmark_times, masks, model_path) where:
            - benchmark_times: [training_time, inference_time] in seconds
            - masks: list of np.ndarray, segmentation masks from test evaluation
            - model_path: str, path to the trained model directory
        """
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
        import tempfile
        import random
        import os
        import tensorflow as tf

        # Determine TensorFlow device string
        if device_context == 'cpu':
            tf_device = '/CPU:0'
        else:
            tf_device = '/GPU:0'
        
        logger.info(f"    Using TensorFlow device: {tf_device}")
        physical_gpus = tf.config.list_physical_devices('GPU')
        logger.debug(f"    Available GPUs: {physical_gpus}")

        # Apply deterministic settings if requested
        deterministic_mode = self.benchmark_params.get('deterministic_mode', False)
        seed = self.benchmark_params.get('seed', 42)

        if deterministic_mode:
            logger.info(f"    Enabling deterministic training with seed={seed}")

            import tensorflow as tf

            # Set environment variables for deterministic behavior
            # Note: These should ideally be set before TF import, but we set them
            # here for cases where TF was already imported
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            os.environ['PYTHONHASHSEED'] = str(seed)

            # Set Python random seed
            random.seed(seed)

            # Set NumPy random seed
            np.random.seed(seed)

            # Set TensorFlow seed
            tf.random.set_seed(seed)

            # Enable TensorFlow deterministic operations (TF 2.8+)
            try:
                tf.config.experimental.enable_op_determinism()
                logger.debug("      TensorFlow op determinism enabled")
            except AttributeError:
                logger.warning("      tf.config.experimental.enable_op_determinism() "
                              "not available (requires TensorFlow 2.8+)")

            logger.debug(f"      Deterministic settings applied:")
            logger.debug(f"        - PYTHONHASHSEED: {seed}")
            logger.debug(f"        - TF_DETERMINISTIC_OPS: 1")
            logger.debug(f"        - TF_CUDNN_DETERMINISTIC: 1")
            logger.debug(f"        - Python random seed: {seed}")
            logger.debug(f"        - NumPy random seed: {seed}")
            logger.debug(f"        - TensorFlow seed: {seed}")

        # Check if we should use pretrained model or train custom model
        use_pretrained = self.benchmark_params.get('use_pretrained_model', False)
        
        if use_pretrained:
            logger.debug(f"    Starting inference benchmark on device: {device_context} (using pretrained model)")
        else:
            logger.debug(f"    Starting training benchmark on device: {device_context}")

        # Get training parameters
        n_epochs = int(self.benchmark_params.get('n_epochs', 1))
        steps_per_epoch = int(self.benchmark_params.get('steps_per_epoch', 10))

        # Create temporary directory for model
        temp_dir = tempfile.mkdtemp(prefix='stardist_benchmark_')
        model_name = 'stardist_benchmark_training'

        start_time = time.time()

        try:
            # Use tf.device context to place operations on correct device
            with tf.device(tf_device):
                logger.info(f"    Creating model on device: {tf_device}")
                if use_pretrained:
                    # Use pretrained StarDist model
                    logger.debug("      Loading pretrained model: 2D_versatile_fluo")
                    model = StarDist2D.from_pretrained('2D_versatile_fluo')
                    train_time = time.time() - start_time
                    logger.debug(f"      Model loaded in {train_time:.2f}s")
                else:
                    # Create a new model for training
                    model = self.model_manager.create_model(
                        name=model_name,
                        basedir=temp_dir
                    )

            # Define augmenter function
            def random_fliprot(img, mask):
                """Random flips and rotations for data augmentation."""
                assert img.ndim >= mask.ndim
                axes = tuple(range(mask.ndim))
                perm = tuple(np.random.permutation(axes))
                img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
                mask = mask.transpose(perm)
                for ax in axes:
                    if np.random.rand() > 0.5:
                        img = np.flip(img, axis=ax)
                        mask = np.flip(mask, axis=ax)
                return img, mask

            def random_intensity_change(img):
                """Random intensity changes for data augmentation."""
                img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
                return img

            def augmenter(x, y):
                """Combined augmenter function."""
                x, y = random_fliprot(x, y)
                x = random_intensity_change(x)
                sig = 0.02 * np.random.uniform(0, 1)
                x = x + sig * np.random.normal(0, 1, x.shape)
                return x, y

            # Normalize training and validation data
            axis_norm = (0, 1)  # Normalize along first two axes
            X_train = [normalize(x, 1, 99.8, axis=axis_norm)
                       for x in self.benchmark_data.train_data]
            Y_train = self.benchmark_data.train_labels

            # Use some training data as validation
            n_val = max(1, len(X_train) // 10)
            X_val = X_train[-n_val:]
            Y_val = Y_train[-n_val:]
            X_train = X_train[:-n_val]
            Y_train = Y_train[:-n_val]

            if not use_pretrained:
                # Train the model with GPU device context
                logger.info(f"    Starting training on device: {tf_device}")
                with tf.device(tf_device):
                    model.train(
                        X_train, Y_train,
                        validation_data=(X_val, Y_val),
                        augmenter=augmenter,
                        epochs=n_epochs,
                        steps_per_epoch=steps_per_epoch
                    )

                train_time = time.time() - start_time
            # else: train_time already set when loading pretrained model

            # Get model path
            model_path = os.path.join(temp_dir, model_name)

            # Run inference on test data
            start_time = time.time()

            # Normalize test data
            X_test = [normalize(x, 1, 99.8, axis=axis_norm)
                      for x in self.benchmark_data.test_data]

            logger.debug(f"    Running inference on {len(X_test)} test images")

            # Run prediction on all test images with device context
            with tf.device(tf_device):
                masks = []
                for i, x in enumerate(X_test):
                    labels, details = model.predict_instances(x)
                    masks.append(labels)
                    n_instances = (len(np.unique(labels)) - 1 
                                   if 0 in np.unique(labels) 
                                   else len(np.unique(labels)))
                    logger.debug(f"      Test image {i}: shape={x.shape}, "
                                f"mask shape={labels.shape}, "
                                f"instances detected={n_instances}")

            inference_time = time.time() - start_time

            logger.debug(f"    Training completed in {train_time:.2f}s, "
                        f"inference in {inference_time:.2f}s")
            logger.debug(f"    Total masks generated: {len(masks)}")

            return [train_time, inference_time], masks, model_path

        except Exception as e:
            # Still return time even if training failed
            benchmark_time = time.time() - start_time
            logger.debug(f"Training failed after {benchmark_time:.4f}s: {e}")
            raise e

    def _check_models_identical(self, model_paths: List[str]) -> bool:
        """
        Check if all trained models are identical by comparing weights.

        For TensorFlow/Keras models, we compare the model weights.

        Args:
            model_paths: List of paths to model directories

        Returns:
            True if all models are identical, False otherwise
        """
        if len(model_paths) < 2:
            logger.debug("Less than 2 models to compare, skipping check")
            return True

        try:
            from stardist.models import StarDist2D
            import tensorflow as tf

            # Load first model as reference
            reference_path = model_paths[0]
            if not Path(reference_path).exists():
                logger.warning(f"Reference model not found: {reference_path}")
                return False

            ref_model = StarDist2D(None, name=Path(reference_path).name,
                                   basedir=str(Path(reference_path).parent))
            ref_weights = ref_model.keras_model.get_weights()

            # Compare all other models
            all_identical = True
            for i, model_path in enumerate(model_paths[1:], start=2):
                if not Path(model_path).exists():
                    logger.warning(f"Model {i} not found: {model_path}")
                    all_identical = False
                    continue

                test_model = StarDist2D(None, name=Path(model_path).name,
                                        basedir=str(Path(model_path).parent))
                test_weights = test_model.keras_model.get_weights()

                # Compare weights
                if len(ref_weights) != len(test_weights):
                    logger.debug(f"Model {i} has different number of weight arrays")
                    all_identical = False
                    continue

                for w1, w2 in zip(ref_weights, test_weights):
                    if not np.allclose(w1, w2, rtol=1e-5, atol=1e-8):
                        logger.debug(f"Model {i} weights differ from reference")
                        all_identical = False
                        break

            return all_identical

        except Exception as e:
            logger.warning(f"Could not compare models: {e}")
            return False

    def _calculate_processed_metrics(self, benchmark_times: List[List[float]]) -> Dict[str, float]:
        """Calculate processed metrics from raw benchmark times."""
        if not benchmark_times:
            return {}

        benchmark_times = np.array(benchmark_times)

        # Extract training times (first) and inference times (second)
        training_times = benchmark_times[:, 0]
        inference_times = benchmark_times[:, 1]

        return {
            # Training time metrics
            'mean_training_time': float(np.mean(training_times)),
            'median_training_time': float(np.median(training_times)),
            'std_training_time': float(np.std(training_times)),
            'min_training_time': float(np.min(training_times)),
            'max_training_time': float(np.max(training_times)),
            'q25_training_time': float(np.percentile(training_times, 25)),
            'q75_training_time': float(np.percentile(training_times, 75)),

            # Inference time metrics
            'mean_inference_time': float(np.mean(inference_times)),
            'median_inference_time': float(np.median(inference_times)),
            'std_inference_time': float(np.std(inference_times)),
            'min_inference_time': float(np.min(inference_times)),
            'max_inference_time': float(np.max(inference_times)),
            'q25_inference_time': float(np.percentile(inference_times, 25)),
            'q75_inference_time': float(np.percentile(inference_times, 75)),

            # Combined metrics
            'total_iterations': len(benchmark_times),
            'mean_total_time': float(np.mean(training_times + inference_times)),
            'median_total_time': float(np.median(training_times +
                                                 inference_times))
        }

    def _calculate_overall_statistics(self, all_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall benchmark statistics."""
        total_tests = 0
        successful_tests = 0
        total_time = 0.0

        for device_key, device_results in all_results.items():
            for test_key, test_result in device_results.items():
                total_tests += 1
                if test_result['success']:
                    successful_tests += 1
                    if test_result['raw_benchmark_times']:
                        # Sum total times (training + inference)
                        for train_time, inference_time in test_result['raw_benchmark_times']:
                            total_time += train_time + inference_time

        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0.0,
            'total_inference_time': total_time,
            'average_inference_time': total_time / successful_tests if successful_tests > 0 else 0.0
        }

    def _convert_results_for_performance_plot(self) -> list:
        """
        Convert all loaded benchmark results into a format suitable for PerformancePlotter.
        """
        if not hasattr(self, 'all_results') or not self.all_results:
            logger.warning("No all_results available for plotting")
            return []

        plotting_data = []

        for result_dict in self.all_results:
            device_results = result_dict.get('results', {}).get('device_results', {})
            flavour = result_dict.get('flavour', 'unknown')

            for device_key, device_data in device_results.items():
                device_parts = device_key.split('_')
                device_type = device_parts[-1]
                device_name = '_'.join(device_parts[:-1])

                for test_key, test_data in device_data.items():
                    if not test_data.get('success', False):
                        continue

                    device_model = test_data.get('device_model', device_name)
                    vendor = self._extract_vendor_from_device(device_name, device_type, device_model)

                    resolution_str = "training"
                    precision = "mixed"

                    device_flavour = test_data.get('arkitekt_flavour', 'unknown')
                    flavour_name = self._create_flavour_name(device_flavour, device_type)

                    processed_metrics = test_data.get('processed_metrics', {})
                    train_time = processed_metrics.get('mean_training_time', 0.0)
                    infer_time = processed_metrics.get('mean_inference_time', 0.0)
                    mean_time = processed_metrics.get('mean_total_time',
                                                      train_time + infer_time)
                    image_size_pixels = 1
                    throughput = 1 / mean_time if mean_time > 0 else 0

                    data_point = {
                        'device_name': device_name,
                        'device_type': device_type,
                        'device_model': device_model,
                        'flavour_name': flavour_name,
                        'vendor': vendor,
                        'flavour': flavour,
                        'execution_time': mean_time,
                        'benchmark_title': 'StarDist Training Performance',
                        'image_size_pixels': image_size_pixels,
                        'inference_time': mean_time,
                        'throughput': throughput,
                        'precision': precision,
                        'resolution': resolution_str,
                        'warmup_iterations': test_data.get('warmup_iterations', 0),
                        'benchmark_iterations': test_data.get('benchmark_iterations', 0),
                        'successful_iterations': test_data.get('successful_iterations', 0),
                        'failed_iterations': test_data.get('failed_iterations', 0),
                        'performance_metrics': {
                            test_key: {
                                'success': True,
                                'resolution': resolution_str,
                                'image_size_pixels': image_size_pixels,
                                'inference_time': mean_time,
                                'throughput_px_per_sec': throughput,
                                'precision_used': precision,
                                'num_cells': 0,
                                **processed_metrics
                            }
                        },
                        'results': {
                            'tests_completed': 1,
                            'tests_failed': 0 if test_data.get('success', False) else 1
                        }
                    }

                    plotting_data.append(data_point)

        logger.debug(f"Converted {len(plotting_data)} benchmark results for plotting")
        return plotting_data

    def _extract_vendor_from_device(self, device_name: str, device_type: str, device_model: str = '') -> str:
        """Extract vendor from device name and type."""
        device_name_lower = device_name.lower()
        device_type_lower = device_type.lower()
        device_model_lower = device_model.lower()

        if 'nvidia' in device_name_lower or device_type_lower == 'cuda' or 'nvidia' in device_model_lower:
            return 'nvidia'
        elif 'intel' in device_name_lower or device_type_lower == 'xpu' or 'intel' in device_model_lower:
            return 'intel'
        elif 'amd' in device_name_lower or device_type_lower in ['hip', 'rocm'] or 'amd' in device_model_lower:
            return 'amd'
        else:
            return 'unknown'

    def _create_flavour_name(self, arkitekt_flavour: str, device_type: str) -> str:
        """Create a descriptive flavour name for the legend."""
        flavour_lower = arkitekt_flavour.lower()
        device_type_lower = device_type.lower()

        if 'nvidia' in flavour_lower or device_type_lower == 'cuda':
            return 'NVIDIA CUDA'
        elif 'intel' in flavour_lower or device_type_lower == 'xpu':
            return 'Intel XPU'
        elif 'amd' in flavour_lower or device_type_lower in ['hip', 'rocm']:
            return 'AMD ROCm'
        else:
            return f'{arkitekt_flavour.title()}'

    def _save_mask(self, mask: np.ndarray, device, test_idx: int = 0) -> str:
        """
        Save inference mask as .npy file alongside results.

        Args:
            mask: Segmentation mask array from test evaluation
            device: Device object with name and type
            test_idx: Index of the test image

        Returns:
            str: Relative filename of saved mask
        """
        try:
            # Create mask directory in the same location as results.json
            flavour = getattr(device, 'arkitekt_flavour', 'unknown')
            mask_dir = (Path(self.config.get('dataset', 'default_dataset')) /
                        self.name / f"{self.timestamp}_{flavour}")
            mask_dir.mkdir(parents=True, exist_ok=True)

            # Create mask filename: device_test{idx}_mask.npy
            device_safe_name = "".join(c for c in device.name
                                       if c.isalnum() or c in (' ', '-', '_'))
            device_safe_name = device_safe_name.rstrip().replace(' ', '_')
            mask_filename = f"{device_safe_name}_test{test_idx}_mask.npy"
            mask_path = mask_dir / mask_filename

            # Log mask info before saving
            if mask is not None:
                unique_vals = np.unique(mask)
                logger.debug(f"      Saving mask: shape={mask.shape}, "
                            f"dtype={mask.dtype}, "
                            f"unique_values={len(unique_vals)} "
                            f"(instances={len(unique_vals)-1 if 0 in unique_vals else len(unique_vals)})")
            else:
                logger.warning(f"      Mask is None, not saving")
                return None

            # Save mask as numpy array
            np.save(mask_path, mask)
            logger.debug(f"      Saved mask: {mask_filename}")

            # Return relative filename for storage in results.json
            return str(mask_filename)

        except Exception as e:
            logger.warning(f"Failed to save mask: {e}")
            return None

    def _evaluate_results(self):
        """
        Evaluate benchmark results by comparing masks and creating visualizations.
        """
        logger.info("Starting mask evaluation and comparison...")

        if not hasattr(self, 'all_results') or not self.all_results:
            logger.warning("No results available for evaluation")
            return

        # Load test data for evaluation (needed for ground truth comparison)
        self._load_test_data_for_evaluation()

        # Group masks by configuration
        mask_groups = self._group_masks_by_config()

        if not mask_groups:
            logger.warning("No masks found for evaluation")
            return

        # Create evaluation plots for each configuration
        for config_key, mask_data in mask_groups.items():
            logger.info(f"Creating evaluation plots for {config_key}")

            # Create ground truth comparisons
            self._create_training_ground_truth_comparison(config_key, mask_data)
            self._create_training_ground_truth_comparison_reduced_cpu(config_key, mask_data)

            # Create device comparison matrix
            self._create_device_comparison_matrix(config_key, mask_data)
            self._create_device_comparison_matrix_reduced_cpu(config_key, mask_data)

        # Create model comparison visualizations
        logger.info("Starting model comparison and analysis...")
        self._create_model_comparison_analysis()

    def _load_test_data_for_evaluation(self):
        """
        Load test data from DSB2018 dataset for evaluation.
        This is needed because evaluation runs in a separate phase from benchmarking.
        """
        from glob import glob
        from tifffile import imread
        from stardist import fill_label_holes

        # Check if data is already loaded
        if (hasattr(self, 'benchmark_data') and
            self.benchmark_data.test_data is not None and
            len(self.benchmark_data.test_data) > 0):
            logger.debug("Test data already loaded")
            return

        # Initialize benchmark_data if needed
        if not hasattr(self, 'benchmark_data'):
            self.benchmark_data = BenchmarkData()

        # Define data directory
        dataset = self.config.get('dataset', 'default_dataset')
        data_dir = Path.cwd() / dataset / self.name / "data"
        test_dir = data_dir / "dsb2018" / "test"

        if not test_dir.exists():
            logger.warning(f"Test data directory not found: {test_dir}")
            return

        # Load test images and masks
        X_test_files = sorted(glob(str(test_dir / "images" / "*.tif")))
        Y_test_files = sorted(glob(str(test_dir / "masks" / "*.tif")))

        if not X_test_files or not Y_test_files:
            logger.warning("No test images or masks found")
            return

        # Load images
        self.benchmark_data.test_data = list(map(imread, X_test_files))
        self.benchmark_data.test_labels = list(map(imread, Y_test_files))

        # Fill label holes (required for StarDist)
        self.benchmark_data.test_labels = [fill_label_holes(y)
                                           for y in self.benchmark_data.test_labels]

        logger.info(f"Loaded {len(self.benchmark_data.test_data)} test images "
                    f"for evaluation")

    def _group_masks_by_config(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group masks by test configuration and test image index for comparison.
        """
        mask_groups = {}

        for result_dict in self.all_results:
            device_results = result_dict.get('results', {}).get('device_results', {})
            flavour = result_dict.get('flavour', 'unknown')
            timestamp = result_dict.get('timestamp', 'unknown')

            for device_key, device_data in device_results.items():
                for test_key, test_data in device_data.items():
                    if (not test_data.get('success', False) or
                            not test_data.get('mask_filenames')):
                        continue

                    mask_filenames = test_data.get('mask_filenames', [])
                    for test_idx, mask_filename in enumerate(mask_filenames):
                        mask_entry = {
                            'device_name': test_data.get('device_name', 'Unknown'),
                            'device_type': test_data.get('device_type', 'unknown'),
                            'device_model': test_data.get('device_model', 'Unknown'),
                            'flavour': flavour,
                            'timestamp': timestamp,
                            'mask_filename': mask_filename,
                            'test_idx': test_idx,
                            'resolution': None,
                            'precision': None,
                            'inference_time': (
                                test_data.get('processed_metrics', {})
                                .get('mean_inference_time', 0.0))
                        }

                        config_key = f"{test_key}test{test_idx}"
                        if config_key not in mask_groups:
                            mask_groups[config_key] = []
                        mask_groups[config_key].append(mask_entry)

        logger.debug(f"Grouped masks into {len(mask_groups)} configurations")
        return mask_groups

    def _create_training_ground_truth_comparison(self, config_key: str, mask_data: List[Dict[str, Any]]):
        """
        Create ground truth comparison visualization for training benchmark.
        """
        if not mask_data:
            return

        try:
            # Extract test index from config_key
            test_idx_str = config_key.split('test')[-1]
            test_idx = int(test_idx_str) if test_idx_str.isdigit() else 0

            # Check if test data is available
            if (not hasattr(self, 'benchmark_data') or
                self.benchmark_data.test_data is None or
                len(self.benchmark_data.test_data) == 0):
                logger.warning("Test data not available for ground truth comparison")
                return

            # Load test data and ground truth
            if test_idx >= len(self.benchmark_data.test_data):
                logger.warning(f"Test index {test_idx} out of range "
                              f"(have {len(self.benchmark_data.test_data)} images)")
                return

            original_image = self.benchmark_data.test_data[test_idx]
            gt_mask = self.benchmark_data.test_labels[test_idx]

            logger.info(f"  Using test image {test_idx} and ground truth")

            # Create comparison plot
            import matplotlib.pyplot as plt

            n_devices = len(mask_data)
            fig, axes = plt.subplots(2, n_devices + 1, figsize=(4*(n_devices + 1), 8))

            # Top-left: Original image
            if len(original_image.shape) == 3:
                if original_image.shape[0] in [1, 2, 3, 4]:
                    axes[0, 0].imshow(original_image[0], cmap='gray')
                else:
                    axes[0, 0].imshow(original_image, cmap='gray')
            else:
                axes[0, 0].imshow(original_image, cmap='gray')
            axes[0, 0].set_title(f"Original Image\nTest {test_idx}", fontsize=10, fontweight='bold')
            axes[0, 0].axis('off')

            # Bottom-left: Ground truth mask
            gt_normalized = self._normalize_instance_mask(gt_mask)
            gt_max = np.max(gt_normalized) if np.max(gt_normalized) > 0 else 1
            axes[1, 0].imshow(gt_normalized, cmap='viridis', vmin=0, vmax=gt_max)
            axes[1, 0].set_title("Ground Truth", fontsize=9, fontweight='bold')
            axes[1, 0].axis('off')

            # Process each device
            for idx, mask_entry in enumerate(mask_data):
                device_mask = self._load_mask(mask_entry)
                if device_mask is None:
                    continue

                col_idx = idx + 1

                # Get device info
                device_flavour = mask_entry.get('flavour', 'unknown')
                device_type = mask_entry['device_type']
                flavour_name = self._create_flavour_name(device_flavour, device_type)
                device_model = mask_entry.get('device_model', mask_entry['device_name'])

                device_title = f"{flavour_name}:\n{device_model}"

                # Top row: Device mask
                device_normalized = self._normalize_instance_mask(device_mask)
                device_max = np.max(device_normalized) if np.max(device_normalized) > 0 else 1
                axes[0, col_idx].imshow(device_normalized, cmap='viridis', vmin=0, vmax=device_max)
                axes[0, col_idx].set_title(device_title, fontsize=10, fontweight='bold')
                axes[0, col_idx].axis('off')

                # Bottom row: Binary difference
                gt_binary = (gt_mask > 0).astype(float)
                device_binary = (device_mask > 0).astype(float)
                diff_binary = (device_binary != gt_binary).astype(float)

                # Create RGB difference image
                diff_rgb = np.ones((diff_binary.shape[0], diff_binary.shape[1], 3))
                diff_rgb[:, :, 1] = 1 - diff_binary
                diff_rgb[:, :, 2] = 1 - diff_binary

                axes[1, col_idx].imshow(diff_rgb)

                # Calculate metrics
                diff_pixels = int(diff_binary.sum())
                total_pixels = int(diff_binary.size)
                diff_percent = (diff_pixels / total_pixels) * 100

                if diff_percent < 0.01 and diff_percent > 0:
                    diff_percent_str = f"{diff_percent:.2e}%"
                else:
                    diff_percent_str = f"{diff_percent:.2f}%"

                gt_cells = len(np.unique(gt_mask)) - 1
                device_cells = len(np.unique(device_mask)) - 1

                axes[1, col_idx].set_title(
                    f"Differences\n({diff_percent_str}, {diff_pixels:,}px)\n"
                    f"GT:{gt_cells} vs Dev:{device_cells}",
                    fontsize=9, fontweight='bold')

                # Add border
                for spine in axes[1, col_idx].spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(3)
                axes[1, col_idx].tick_params(which='both', length=0)
                axes[1, col_idx].set_xticks([])
                axes[1, col_idx].set_yticks([])

            plt.suptitle(f'StarDist Training Ground Truth Comparison - Test Image {test_idx}\n'
                        f'Top: Instance Masks | Bottom: Binary Mask Differences (White=Match, Red=Differ)',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Save plot
            self._save_evaluation_plot(fig, f'ground_truth_comparison_{config_key}')
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create ground truth comparison for {config_key}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _create_training_ground_truth_comparison_reduced_cpu(self, config_key: str, mask_data: List[Dict[str, Any]]):
        """
        Create ground truth comparison with reduced CPU representation.
        """
        if not mask_data:
            return

        # Filter to reduce CPU devices
        filtered_data = self._reduce_cpu_devices(mask_data)

        if len(filtered_data) == len(mask_data):
            logger.debug("No CPU reduction needed, skipping reduced version")
            return

        try:
            # Extract test index from config_key
            test_idx_str = config_key.split('test')[-1]
            test_idx = int(test_idx_str) if test_idx_str.isdigit() else 0

            # Check if test data is available
            if (not hasattr(self, 'benchmark_data') or
                self.benchmark_data.test_data is None or
                len(self.benchmark_data.test_data) == 0):
                logger.warning("Test data not available for reduced CPU comparison")
                return

            # Load test data and ground truth
            if test_idx >= len(self.benchmark_data.test_data):
                logger.warning(f"Test index {test_idx} out of range")
                return

            original_image = self.benchmark_data.test_data[test_idx]
            gt_mask = self.benchmark_data.test_labels[test_idx]

            # Create comparison plot
            import matplotlib.pyplot as plt

            n_devices = len(filtered_data)
            fig, axes = plt.subplots(2, n_devices + 1, figsize=(4*(n_devices + 1), 8))

            # Top-left: Original image
            if len(original_image.shape) == 3:
                if original_image.shape[0] in [1, 2, 3, 4]:
                    axes[0, 0].imshow(original_image[0], cmap='gray')
                else:
                    axes[0, 0].imshow(original_image, cmap='gray')
            else:
                axes[0, 0].imshow(original_image, cmap='gray')
            axes[0, 0].set_title(f"Original Image\nTest {test_idx}", fontsize=10, fontweight='bold')
            axes[0, 0].axis('off')

            # Bottom-left: Ground truth mask
            gt_normalized = self._normalize_instance_mask(gt_mask)
            gt_max = np.max(gt_normalized) if np.max(gt_normalized) > 0 else 1
            axes[1, 0].imshow(gt_normalized, cmap='viridis', vmin=0, vmax=gt_max)
            axes[1, 0].set_title("Ground Truth", fontsize=9, fontweight='bold')
            axes[1, 0].axis('off')

            # Process each device
            for idx, mask_entry in enumerate(filtered_data):
                device_mask = self._load_mask(mask_entry)
                if device_mask is None:
                    continue

                col_idx = idx + 1

                # Get device info
                device_flavour = mask_entry.get('flavour', 'unknown')
                device_type = mask_entry['device_type']
                device_model = mask_entry.get('device_model', mask_entry['device_name'])

                # Remove flavour for CPU in reduced version
                if device_type.lower() == 'cpu':
                    device_title = device_model
                else:
                    flavour_name = self._create_flavour_name(device_flavour, device_type)
                    device_title = f"{flavour_name}:\n{device_model}"

                # Top row: Device mask
                device_normalized = self._normalize_instance_mask(device_mask)
                device_max = np.max(device_normalized) if np.max(device_normalized) > 0 else 1
                axes[0, col_idx].imshow(device_normalized, cmap='viridis', vmin=0, vmax=device_max)
                axes[0, col_idx].set_title(device_title, fontsize=10, fontweight='bold')
                axes[0, col_idx].axis('off')

                # Bottom row: Binary difference
                gt_binary = (gt_mask > 0).astype(float)
                device_binary = (device_mask > 0).astype(float)
                diff_binary = (device_binary != gt_binary).astype(float)

                # Create RGB difference image
                diff_rgb = np.ones((diff_binary.shape[0], diff_binary.shape[1], 3))
                diff_rgb[:, :, 1] = 1 - diff_binary
                diff_rgb[:, :, 2] = 1 - diff_binary

                axes[1, col_idx].imshow(diff_rgb)

                # Calculate metrics
                diff_pixels = int(diff_binary.sum())
                total_pixels = int(diff_binary.size)
                diff_percent = (diff_pixels / total_pixels) * 100

                if diff_percent < 0.01 and diff_percent > 0:
                    diff_percent_str = f"{diff_percent:.2e}%"
                else:
                    diff_percent_str = f"{diff_percent:.2f}%"

                gt_cells = len(np.unique(gt_mask)) - 1
                device_cells = len(np.unique(device_mask)) - 1

                axes[1, col_idx].set_title(
                    f"Differences\n({diff_percent_str}, {diff_pixels:,}px)\n"
                    f"GT:{gt_cells} vs Dev:{device_cells}",
                    fontsize=9, fontweight='bold')

                # Add border
                for spine in axes[1, col_idx].spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(3)
                axes[1, col_idx].tick_params(which='both', length=0)
                axes[1, col_idx].set_xticks([])
                axes[1, col_idx].set_yticks([])

            plt.suptitle(f'StarDist Training Ground Truth Comparison (Reduced CPU) - Test Image {test_idx}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Save plot
            self._save_evaluation_plot(fig, f'ground_truth_comparison_reduced_cpu_{config_key}')
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create reduced CPU ground truth comparison: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _create_device_comparison_matrix(self, config_key: str, mask_data: List[Dict[str, Any]]):
        """
        Create device-to-device comparison matrix.
        """
        if len(mask_data) < 2:
            logger.debug(f"Not enough devices for comparison matrix in {config_key}")
            return

        try:
            # Load all masks and normalize them
            loaded_masks = []
            device_info = []

            for mask_entry in mask_data:
                mask = self._load_mask(mask_entry)
                if mask is not None:
                    normalized_mask = self._normalize_instance_mask(mask)
                    loaded_masks.append((mask, normalized_mask))

                    device_flavour = mask_entry.get('flavour', 'unknown')
                    device_type = mask_entry['device_type']
                    flavour_name = self._create_flavour_name(device_flavour, device_type)
                    device_model = mask_entry.get('device_model', mask_entry['device_name'])

                    device_info.append({
                        'flavour_name': flavour_name,
                        'device_model': device_model,
                        'title': f"{flavour_name}:\n{device_model}"
                    })

            if len(loaded_masks) < 2:
                return

            # Create comparison matrix
            import matplotlib.pyplot as plt

            n_devices = len(loaded_masks)
            fig, axes = plt.subplots(n_devices, n_devices, figsize=(4*n_devices, 4*n_devices))

            if n_devices == 1:
                axes = [[axes]]
            elif n_devices == 2:
                axes = axes.reshape(2, 2)

            for i in range(n_devices):
                for j in range(n_devices):
                    original_mask_i, normalized_mask_i = loaded_masks[i]
                    original_mask_j, normalized_mask_j = loaded_masks[j]

                    if i == j:
                        mask_max_instances = np.max(normalized_mask_i) if np.max(normalized_mask_i) > 0 else 1
                        axes[i][j].imshow(normalized_mask_i, cmap='viridis', vmin=0, vmax=mask_max_instances)
                        axes[i][j].set_title(device_info[i]['title'], fontsize=10, fontweight='bold')
                    elif i > j:
                        binary_i = (original_mask_i > 0).astype(float)
                        binary_j = (original_mask_j > 0).astype(float)
                        diff_binary = (binary_i != binary_j).astype(float)

                        diff_rgb = np.ones((diff_binary.shape[0], diff_binary.shape[1], 3))
                        diff_rgb[:, :, 1] = 1 - diff_binary
                        diff_rgb[:, :, 2] = 1 - diff_binary

                        axes[i][j].imshow(diff_rgb)

                        diff_pixels = int(diff_binary.sum())
                        total_pixels = int(diff_binary.size)
                        diff_percent = (diff_pixels / total_pixels) * 100

                        if diff_percent < 0.01 and diff_percent > 0:
                            diff_percent_str = f"{diff_percent:.2e}%"
                        else:
                            diff_percent_str = f"{diff_percent:.2f}%"

                        cells_i = len(np.unique(original_mask_i)) - 1
                        cells_j = len(np.unique(original_mask_j)) - 1

                        axes[i][j].set_title(f"Binary Mask Differences\n({diff_percent_str}, {diff_pixels:,} pixels)\n{cells_i} vs {cells_j} cells",
                                           fontsize=9, fontweight='bold')

                        for spine in axes[i][j].spines.values():
                            spine.set_visible(True)
                            spine.set_color('black')
                            spine.set_linewidth(3)
                        axes[i][j].tick_params(which='both', length=0)
                    else:
                        axes[i][j].set_visible(False)

                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])

            test_idx = config_key.split('test')[-1] if 'test' in config_key else '0'
            plt.suptitle(f'StarDist Training Device-to-Device Comparison Matrix - Test Image {test_idx}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            self._save_evaluation_plot(fig, f'device_comparison_matrix_{config_key}')
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create device comparison matrix for {config_key}: {e}")

    def _create_device_comparison_matrix_reduced_cpu(self, config_key: str, mask_data: List[Dict[str, Any]]):
        """
        Create device-to-device comparison matrix with reduced CPU representation.
        """
        if len(mask_data) < 2:
            return

        try:
            filtered_mask_data = self._reduce_cpu_devices(mask_data)

            if len(filtered_mask_data) == len(mask_data):
                logger.debug(f"No CPU reduction needed for {config_key}")
                return

            if len(filtered_mask_data) < 2:
                return

            # Load all masks
            loaded_masks = []
            device_info = []

            for mask_entry in filtered_mask_data:
                mask = self._load_mask(mask_entry)
                if mask is not None:
                    normalized_mask = self._normalize_instance_mask(mask)
                    loaded_masks.append((mask, normalized_mask))

                    device_flavour = mask_entry.get('flavour', 'unknown')
                    device_type = mask_entry['device_type']
                    device_model = mask_entry.get('device_model', mask_entry['device_name'])

                    if device_type.lower() == 'cpu':
                        title = device_model
                    else:
                        flavour_name = self._create_flavour_name(device_flavour, device_type)
                        title = f"{flavour_name}:\n{device_model}"

                    device_info.append({'title': title})

            if len(loaded_masks) < 2:
                return

            import matplotlib.pyplot as plt

            n_devices = len(loaded_masks)
            fig, axes = plt.subplots(n_devices, n_devices, figsize=(4*n_devices, 4*n_devices))

            if n_devices == 2:
                axes = axes.reshape(2, 2)

            for i in range(n_devices):
                for j in range(n_devices):
                    original_mask_i, normalized_mask_i = loaded_masks[i]
                    original_mask_j, normalized_mask_j = loaded_masks[j]

                    if i == j:
                        mask_max = np.max(normalized_mask_i) if np.max(normalized_mask_i) > 0 else 1
                        axes[i][j].imshow(normalized_mask_i, cmap='viridis', vmin=0, vmax=mask_max)
                        axes[i][j].set_title(device_info[i]['title'], fontsize=10, fontweight='bold')
                    elif i > j:
                        binary_i = (original_mask_i > 0).astype(float)
                        binary_j = (original_mask_j > 0).astype(float)
                        diff_binary = (binary_i != binary_j).astype(float)

                        diff_rgb = np.ones((diff_binary.shape[0], diff_binary.shape[1], 3))
                        diff_rgb[:, :, 1] = 1 - diff_binary
                        diff_rgb[:, :, 2] = 1 - diff_binary

                        axes[i][j].imshow(diff_rgb)

                        diff_pixels = int(diff_binary.sum())
                        diff_percent = (diff_pixels / diff_binary.size) * 100

                        if diff_percent < 0.01 and diff_percent > 0:
                            diff_str = f"{diff_percent:.2e}%"
                        else:
                            diff_str = f"{diff_percent:.2f}%"

                        cells_i = len(np.unique(original_mask_i)) - 1
                        cells_j = len(np.unique(original_mask_j)) - 1

                        axes[i][j].set_title(f"Differences\n({diff_str})\n{cells_i} vs {cells_j} cells",
                                           fontsize=9, fontweight='bold')

                        for spine in axes[i][j].spines.values():
                            spine.set_visible(True)
                            spine.set_color('black')
                            spine.set_linewidth(3)
                    else:
                        axes[i][j].set_visible(False)

                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])

            test_idx = config_key.split('test')[-1] if 'test' in config_key else '0'
            plt.suptitle(f'StarDist Training Device Comparison (Reduced CPU) - Test Image {test_idx}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            self._save_evaluation_plot(fig, f'device_comparison_matrix_reduced_cpu_{config_key}')
            plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create reduced CPU comparison matrix: {e}")

    def _reduce_cpu_devices(self, mask_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reduce CPU devices to only one representative to avoid redundancy.
        """
        filtered_data = []
        cpu_found = False

        for mask_entry in mask_data:
            device_type = mask_entry.get('device_type', '').lower()

            if device_type == 'cpu':
                if not cpu_found:
                    filtered_data.append(mask_entry)
                    cpu_found = True
            else:
                filtered_data.append(mask_entry)

        return filtered_data

    def _normalize_instance_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Normalize instance segmentation mask for consistent visualization.
        """
        if mask is None:
            return mask

        normalized_mask = np.zeros_like(mask)

        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]

        if len(unique_ids) == 0:
            return normalized_mask

        centroids = []
        for instance_id in unique_ids:
            y_coords, x_coords = np.where(mask == instance_id)
            if len(y_coords) > 0:
                centroid_y = np.mean(y_coords)
                centroid_x = np.mean(x_coords)
                centroids.append((centroid_y, centroid_x, instance_id))

        centroids.sort(key=lambda c: (c[0], c[1]))

        for new_id, (_, _, original_id) in enumerate(centroids, start=1):
            normalized_mask[mask == original_id] = new_id

        return normalized_mask

    def _load_mask(self, mask_entry: Dict[str, Any]) -> np.ndarray:
        """Load mask from file based on mask entry."""
        try:
            flavour = mask_entry['flavour']
            timestamp = mask_entry['timestamp']
            mask_filename = mask_entry['mask_filename']

            mask_dir = Path(self.config.get('dataset', 'default_dataset')) / self.name / f"{timestamp}_{flavour}"
            mask_path = mask_dir / mask_filename

            logger.debug(f"Loading mask from: {mask_path}")

            if mask_path.exists():
                loaded_data = np.load(mask_path, allow_pickle=True)
                logger.debug(f"Loaded mask dtype: {loaded_data.dtype}, "
                            f"shape: {loaded_data.shape if hasattr(loaded_data, 'shape') else 'N/A'}")

                if loaded_data.dtype == object:
                    if loaded_data.shape == ():
                        actual_mask = loaded_data.item()
                        if isinstance(actual_mask, list) and len(actual_mask) > 0:
                            logger.debug(f"Extracted mask from list, "
                                        f"shape: {actual_mask[0].shape}")
                            return actual_mask[0]
                        else:
                            return actual_mask
                    else:
                        return loaded_data
                else:
                    # Check if mask is empty (all zeros)
                    unique_vals = np.unique(loaded_data)
                    logger.debug(f"Mask unique values: {unique_vals[:10]}... "
                                f"(total: {len(unique_vals)})")
                    return loaded_data
            else:
                logger.warning(f"Mask file not found: {mask_path}")
                return None

        except Exception as e:
            logger.warning(f"Failed to load mask: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _create_model_comparison_analysis(self):
        """
        Create comprehensive model comparison analysis for TensorFlow/Keras models.
        """
        logger.info("Creating model comparison analysis...")

        model_info = self._collect_model_info()

        if len(model_info) < 2:
            logger.warning("Not enough models for comparison (need at least 2)")
            return

        self._create_model_comparison_matrix(model_info)
        self._create_model_statistics_table(model_info)

    def _collect_model_info(self) -> List[Dict[str, Any]]:
        """Collect model paths and metadata from all results."""
        model_info = []

        for result_dict in self.all_results:
            device_results = result_dict.get('results', {}).get('device_results', {})
            flavour = result_dict.get('flavour', 'unknown')
            timestamp = result_dict.get('timestamp', 'unknown')

            for device_key, device_data in device_results.items():
                for test_key, test_data in device_data.items():
                    if not test_data.get('success', False):
                        continue

                    model_paths = test_data.get('model_paths', [])
                    if not model_paths:
                        continue

                    model_path = model_paths[0]

                    device_flavour = test_data.get('arkitekt_flavour', 'unknown')
                    device_type = test_data.get('device_type', 'unknown')
                    device_name = test_data.get('device_name', 'Unknown')
                    device_model = test_data.get('device_model', device_name)

                    flavour_name = self._create_flavour_name(device_flavour, device_type)

                    model_info.append({
                        'model_path': model_path,
                        'device_name': device_name,
                        'device_type': device_type,
                        'device_model': device_model,
                        'flavour': flavour,
                        'flavour_name': flavour_name,
                        'timestamp': timestamp,
                        'title': f"{flavour_name}:\n{device_model}"
                    })

        logger.debug(f"Collected {len(model_info)} models for comparison")
        return model_info

    def _create_model_comparison_matrix(self, model_info: List[Dict[str, Any]]):
        """
        Create a similarity matrix comparing all TensorFlow/Keras models.
        """
        import matplotlib.pyplot as plt

        n_models = len(model_info)
        if n_models < 2:
            return

        logger.info(f"Creating model comparison matrix for {n_models} models...")

        try:
            from stardist.models import StarDist2D

            # Load all models
            models = []
            labels = []
            for info in model_info:
                try:
                    model_path = Path(info['model_path'])
                    model = StarDist2D(None, name=model_path.name,
                                       basedir=str(model_path.parent))
                    models.append(model.keras_model.get_weights())
                    labels.append(info['title'])
                except Exception as e:
                    logger.warning(f"Failed to load model {info['model_path']}: {e}")
                    return

            # Create comparison matrices
            l2_distance_matrix = np.zeros((n_models, n_models))
            cosine_sim_matrix = np.zeros((n_models, n_models))

            for i in range(n_models):
                for j in range(n_models):
                    if i == j:
                        l2_distance_matrix[i, j] = 0
                        cosine_sim_matrix[i, j] = 1.0
                    else:
                        metrics = self._compare_two_keras_models(models[i], models[j])
                        l2_distance_matrix[i, j] = metrics['l2_distance']
                        cosine_sim_matrix[i, j] = metrics['cosine_similarity']

            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: L2 Distance
            im1 = axes[0].imshow(l2_distance_matrix, cmap='YlOrRd', aspect='auto')
            axes[0].set_title('L2 Distance\n(Lower = More Similar)', fontsize=12, fontweight='bold')
            axes[0].set_xticks(range(n_models))
            axes[0].set_yticks(range(n_models))
            axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            axes[0].set_yticklabels(labels, fontsize=8)
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            for i in range(n_models):
                for j in range(n_models):
                    axes[0].text(j, i, f'{l2_distance_matrix[i, j]:.2e}',
                                ha="center", va="center", color="black", fontsize=7)

            # Plot 2: Cosine Similarity
            im2 = axes[1].imshow(cosine_sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            axes[1].set_title('Cosine Similarity\n(Higher = More Similar)', fontsize=12, fontweight='bold')
            axes[1].set_xticks(range(n_models))
            axes[1].set_yticks(range(n_models))
            axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            axes[1].set_yticklabels(labels, fontsize=8)
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            for i in range(n_models):
                for j in range(n_models):
                    axes[1].text(j, i, f'{cosine_sim_matrix[i, j]:.6f}',
                                ha="center", va="center", color="black", fontsize=7)

            plt.suptitle('Model Weight Similarity Comparison Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()

            self._save_evaluation_plot(fig, 'model_comparison_matrix')
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Failed to create model comparison matrix: {e}")

    def _compare_two_keras_models(self, weights1: list, weights2: list) -> Dict[str, float]:
        """
        Compare two Keras model weight lists using multiple metrics.
        """
        if len(weights1) != len(weights2):
            return {'l2_distance': float('inf'), 'cosine_similarity': 0.0}

        # Flatten all weights
        vec1 = np.concatenate([w.flatten() for w in weights1])
        vec2 = np.concatenate([w.flatten() for w in weights2])

        # L2 distance
        l2_dist = np.linalg.norm(vec1 - vec2)

        # Cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        else:
            cosine_sim = 0.0

        return {
            'l2_distance': float(l2_dist),
            'cosine_similarity': float(cosine_sim)
        }

    def _create_model_statistics_table(self, model_info: List[Dict[str, Any]]):
        """
        Create a detailed statistics table for all models.
        """
        import matplotlib.pyplot as plt

        logger.info("Creating model statistics table...")

        stats_data = []

        for info in model_info:
            try:
                from stardist.models import StarDist2D

                model_path = Path(info['model_path'])
                model = StarDist2D(None, name=model_path.name,
                                   basedir=str(model_path.parent))

                weights = model.keras_model.get_weights()

                all_weights = np.concatenate([w.flatten() for w in weights])
                n_params = len(all_weights)
                n_layers = len(weights)

                # Get directory size
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())

                stats = {
                    'Device': info['title'].replace('\n', ' '),
                    'Layers': n_layers,
                    'Parameters': f"{n_params:,}",
                    'Mean': f"{np.mean(all_weights):.6f}",
                    'Std': f"{np.std(all_weights):.6f}",
                    'Min': f"{np.min(all_weights):.6f}",
                    'Max': f"{np.max(all_weights):.6f}",
                    'Size': f"{total_size / 1024 / 1024:.2f} MB"
                }
                stats_data.append(stats)

            except Exception as e:
                logger.warning(f"Failed to get statistics for {info['model_path']}: {e}")

        if not stats_data:
            logger.warning("No model statistics collected")
            return

        # Create figure with table
        fig, ax = plt.subplots(figsize=(14, len(stats_data) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')

        columns = list(stats_data[0].keys())
        rows = [[stats[col] for col in columns] for stats in stats_data]

        table = ax.table(cellText=rows, colLabels=columns, cellLoc='center',
                        loc='center', bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Header styling
        for i in range(len(columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(rows) + 1):
            for j in range(len(columns)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#E7E6E6')
                else:
                    cell.set_facecolor('#FFFFFF')

        plt.title('Model Statistics Summary', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        self._save_evaluation_plot(fig, 'model_statistics_table')
        plt.close(fig)

    def _save_evaluation_plot(self, fig, plot_name: str):
        """Save evaluation plot alongside other benchmark plots."""
        try:
            from datetime import datetime

            dataset = self.config.get('dataset', 'default_dataset')
            plot_dir = Path(dataset) / self.name / f"{self.timestamp}_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Save as both PNG and SVG
            png_path = plot_dir / f"{plot_name}.png"
            svg_path = plot_dir / f"{plot_name}.svg"

            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(svg_path, bbox_inches='tight')

            logger.info(f"  Evaluation plot saved: {plot_name}")

        except Exception as e:
            logger.error(f"Failed to save evaluation plot {plot_name}: {e}")

    def _evaluate_benchmark(self) -> None:
        """
        Override base class evaluation to add mask comparison and visualization.
        """
        logger.info("Evaluating StarDist benchmark results with mask comparison...")

        # First create performance plots
        from ..core.visualizations import PerformancePlotter

        # Convert real benchmark results for plotting
        plotting_data = self._convert_results_for_performance_plot()

        try:
            # Create performance plotter with real data
            plotter = PerformancePlotter(plotting_data)

            # Create plot path with evaluation timestamp
            from datetime import datetime
            dataset = self.config.get('dataset', 'default_dataset')
            plot_path = f"{dataset}/{self.name}/{self.timestamp}_plots/performance_benchmark"

            plotter.store_plot(plot_path)
            logger.info(f"Performance plot saved to: {plot_path}")

        except ImportError as e:
            logger.warning(f"Could not create performance plot: {e}")
        except Exception as e:
            logger.error(f"Error creating performance plot: {e}")

        # Then create mask evaluation plots
        self._evaluate_results()
