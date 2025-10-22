"""
Artificial test benchmark for testing the bench-ARK framework with pytorch.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
from .pytorch_base import PyTorchBenchmark

logger = logging.getLogger(__name__)


class ModelManager:
    """Simplified version of ModelManager for Cellpose model loading and device assignment"""
    
    def __init__(self):
        self.model = None
        self.current_device = None
    
    def load_model(self, device_context: str) -> None:
        """Load Cellpose model for the specified device"""
        logger.debug(f"Loading Cellpose model for device: {device_context}")
        
        try:
            from cellpose import models
            
            # Determine GPU settings
            use_gpu = device_context != 'cpu'
            
            # Load model with appropriate GPU setting
            if use_gpu:
                logger.debug(f"  Loading model with GPU support for {device_context}")
                self.model = models.CellposeModel(gpu=True)
            else:
                logger.debug(f"  Loading model for CPU")
                self.model = models.CellposeModel(gpu=False)
            
            # Try to move model to specific device if using GPU
            if use_gpu and hasattr(self.model, 'net') and self.model.net is not None:
                self._move_model_to_device(device_context)
            
            self.current_device = device_context
            logger.debug(f"  Model loaded successfully for {device_context}")
            
        except ImportError as e:
            raise ImportError(f"Cellpose not available: {e}")
    
    def _move_model_to_device(self, device_context: str) -> None:
        """Attempt to move model to specific device"""
        try:
            if device_context.startswith('cuda'):
                self.model.net = self.model.net.to(device_context)
                logger.debug(f"  Model moved to {device_context}")
            elif device_context.startswith('xpu'):
                self.model.net = self.model.net.to('xpu')
                logger.debug(f"  Model moved to Intel XPU")
            elif device_context.startswith('hip'):
                # AMD GPU using HIP, but PyTorch uses CUDA API
                self.model.net = self.model.net.to(device_context.replace('hip', 'cuda'))
                logger.debug(f"  Model moved to AMD GPU via CUDA API")
        except Exception as e:
            logger.warning(f"  Could not move model to {device_context}: {e}")
            logger.debug(f"  Model will use default device")
    
    def is_loaded_for_device(self, device_context: str) -> bool:
        """Check if model is already loaded for this device"""
        return self.model is not None and self.current_device == device_context
    
    def eval(self, image, diameter=None, channels=None, flow_threshold=0.4, cellprob_threshold=0.0):
        """Run Cellpose inference"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.model.eval(
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )

class BenchmarkData:
    def __init__(self):
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None


class CellposeTraining(PyTorchBenchmark):
    """
    A simple artificial benchmark for testing the bench-ARK framework.
    
    This benchmark performs basic mathematical operations to simulate
    workload and test device detection, benchmark execution, and result handling.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        logger.debug("CellposeArtificialInference initialization starting...")
        super().__init__(name, config)
        logger.debug("CellposeArtificialInference initialization completed.")
        self.model_manager = ModelManager()
        self.benchmark_data = BenchmarkData()
        self.benchmark_params = self.config.get('benchmarks', {}).get('parameters', {}).get(self.name, {})

    def _run_benchmark(self) -> Dict[str, Any]:
        """

        """
        logger.info(f"Starting Cellpose Training Benchmark")

        self.benchmark_data.train_data, self.benchmark_data.train_labels, self.benchmark_data.test_data, self.benchmark_data.test_labels = self._download_and_prepare_training_data()

        # Get benchmark parameters from config
        warmup_iterations = self.benchmark_params.get('warmup_iterations', 1)
        num_iterations = self.benchmark_params.get('num_iterations', 1)
        n_epochs = int(self.config.get('n_epochs', 1))
        learning_rate = float(self.config.get('learning_rate', 0.001))
        weight_decay = float(self.config.get('weight_decay', 0.0001))
        batch_size = int(self.config.get('batch_size', 8))

        import cellpose
        logger.debug(f"Cellpose path: {cellpose.__file__}")
        logger.debug(f"Benchmark parameters:")
        logger.debug(f"  Warmup iterations: {warmup_iterations}")
        logger.debug(f"  Benchmark iterations: {num_iterations}")
        logger.debug(f"  n_epochs: {n_epochs}")
        logger.debug(f"  Learning rate: {learning_rate}")
        logger.debug(f"  Weight decay: {weight_decay}")
        logger.debug(f"  Batch size: {batch_size}")

        # Run benchmarks for each device
        all_results = {}
        devices = self.device_manager._list_available_devices()
        
        for device in devices:
            logger.info(f"\n--- Testing Device: {device.name} ({device.type}) ---")
            
            device_results = {}
            
            # Setup model for this device
            device_context = self._setup_device_context(device)
            self.model_manager = self._load_model_for_device(self.model_manager, device_context)
            
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
            
            logger.debug(f"      Running {num_iterations} benchmark "
                         "iterations...")
            for iteration in range(num_iterations):
                try:
                    benchmark_time, mask, model_path = (
                        self._run_single_benchmark(device_context))
                    benchmark_times.append(benchmark_time)
                    masks_collected.append(mask)
                    
                    # Copy model to results folder with device name and
                    # iteration number
                    model_name = (f"{device_safe_name}_"
                                  f"cellpose_training_model_{iteration + 1}")
                    stored_model_path = models_dir / model_name
                    import shutil
                    shutil.copy2(model_path, stored_model_path)
                    model_paths.append(str(stored_model_path))
                    
                    successful_iterations += 1
                except Exception as e:
                    logger.warning(f"Benchmark iteration {iteration+1} "
                                   f"failed: {e}")
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
            test_key = "cellpose_training "
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
                'mask_filenames': mask_filenames,  # List of filenames for all test images
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
                logger.info(f"      ✓ Completed: "
                            f"avg_train={avg_training_time:.4f}s, "
                            f"avg_inference={avg_inference_time:.4f}s, "
                            f"runs={len(benchmark_times)}/{num_iterations}")
            else:
                logger.warning("      ✗ All iterations failed")
            
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
                # 'data_directory': str(data_dir),
                'benchmark_parameters': str(self.benchmark_params)
            }
        }
        
        logger.info("\n✓ Benchmark completed successfully")
        logger.info(f"  Devices tested: {len(devices)}")
        total_config = (
            final_results['benchmark_metadata']['total_configurations'])
        logger.info(f"  Total configurations: {total_config}")
        
        return final_results

    def _download_and_prepare_training_data(self):
        from cellpose import utils, io
        from pathlib import Path
        import zipfile

        # Define data directory for this benchmark
        dataset = self.config.get('dataset', 'default_dataset')
        data_dir = Path.cwd() / dataset / self.name / "data"
        
        # Define paths for training data
        train_dir = data_dir / "human_in_the_loop" / "train"
        test_dir = data_dir / "human_in_the_loop" / "test"
        
        # Check if data already exists
        if train_dir.exists() and test_dir.exists():
            logger.info("Training data exists, loading from data folder...")
        else:
            logger.info("Training data not found, downloading...")
            
            # Create data directory if it doesn't exist
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Download the training data
            url = ("https://drive.google.com/uc?"
                   "id=1HXpLczf7TPCdI1yZY5KV3EkdWzRrgvhQ")
            zip_path = data_dir / "human_in_the_loop.zip"
            utils.download_url_to_file(url, str(zip_path))
            
            # Extract to data directory
            logger.info("Extracting training data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Clean up zip file after extraction
            zip_path.unlink()
            logger.info("Training data downloaded and extracted successfully")
        
        # Load training and test data
        masks_ext = "_seg.npy"
        output = io.load_train_test_data(str(train_dir), str(test_dir),
                                         mask_filter=masks_ext)
        train_data, train_labels, _, test_data, test_labels, _ = output
        logger.debug(f"Loaded {len(train_data)} training images and "
                     f"{len(test_data)} test images.")

        return train_data, train_labels, test_data, test_labels

            
    def _setup_device_context(self, device) -> str:
        """Setup device context string for PyTorch."""
        if device.type.lower() == 'cpu':
            return 'cpu'
        elif device.type.lower() == 'cuda':
            return f'cuda:{device.id}'
        elif device.type.lower() == 'xpu':
            return f'xpu:{device.id}'
        elif device.type.lower() in ['rocm', 'hip']:
            return f'cuda:{device.id}'  # AMD GPUs use CUDA API in PyTorch
        else:
            return 'cpu'  # Fallback
    
    def _load_model_for_device(self, model_manager, device_context: str):
        """Load Cellpose model for specific device."""
        logger.debug(f"Setting up model for device: {device_context}")
        
        # Check if model is already loaded for this device
        if not model_manager.is_loaded_for_device(device_context):
            model_manager.load_model(device_context)
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
            - masks: np.ndarray, segmentation mask from test evaluation
            - model_path: str, path to the trained model file
        """
        from cellpose import train, models
        # import torch
        import random
        import os
        
        # Apply deterministic settings if requested
        deterministic = self.benchmark_params.get('deterministic', False)
        seed = self.benchmark_params.get('seed', 42)
        
        if deterministic:
            logger.info(f"    Enabling deterministic training with seed={seed}")
            
            # Set CUBLAS environment variable for deterministic CUDA operations
            # This must be set before any CUDA operations are performed
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            logger.debug("      CUBLAS_WORKSPACE_CONFIG=:4096:8 set for "
                        "deterministic CUDA operations")
            
            # Set Python random seed
            random.seed(seed)
            
            # Set NumPy random seed
            np.random.seed(seed)
            
            # Set PyTorch seeds
            self.torch.manual_seed(seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed(seed)
                self.torch.cuda.manual_seed_all(seed)
            
            # Enable deterministic operations
            self.torch.backends.cudnn.deterministic = True
            self.torch.backends.cudnn.benchmark = False
            
            # For PyTorch 1.8+, use deterministic algorithms
            if hasattr(self.torch, 'use_deterministic_algorithms'):
                try:
                    # Use warn_only=True to allow GPU operations that don't have
                    # deterministic implementations (like upsample_linear1d_backward)
                    deterministic_warn_only = True
                    self.torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
                    logger.debug(
                        "      torch.use_deterministic_algorithms(True, "
                        "warn_only="+str(deterministic_warn_only)+") enabled"
                    )
                except TypeError:
                    # Older PyTorch versions don't support warn_only parameter
                    try:
                        self.torch.use_deterministic_algorithms(True)
                        logger.debug(
                            "      torch.use_deterministic_algorithms(True) "
                            "enabled (warn_only not supported)"
                        )
                    except Exception as e:
                        logger.warning(
                            f"      Could not enable deterministic "
                            f"algorithms: {e}"
                        )
                except Exception as e:
                    logger.warning(
                        f"      Could not enable deterministic algorithms: {e}"
                    )
            
            logger.debug(f"      Deterministic settings applied:")
            logger.debug(f"        - Python random seed: {seed}")
            logger.debug(f"        - NumPy random seed: {seed}")
            logger.debug(f"        - PyTorch manual seed: {seed}")
            logger.debug(f"        - CUDA seeds: {seed}")
            logger.debug(f"        - cudnn.deterministic: True")
            logger.debug(f"        - cudnn.benchmark: False")
        
        logger.debug(f"    Starting training benchmark on device: {device_context}")
        
        # Verify the model is on the correct device before training
        if hasattr(self.model_manager.model, 'net') and self.model_manager.model.net is not None:
            try:
                # Get the device of the first parameter
                model_device = next(self.model_manager.model.net.parameters()).device
                logger.info(f"    Model is currently on device: {model_device}")
                
                # If model is not on the expected device, move it
                if str(model_device) != device_context and device_context != 'cpu':
                    logger.warning(f"    Model device mismatch! Expected {device_context}, but model is on {model_device}")
                    logger.info(f"    Moving model to {device_context}...")
                    self.model_manager._move_model_to_device(device_context)
                    model_device_after = next(self.model_manager.model.net.parameters()).device
                    logger.info(f"    Model is now on device: {model_device_after}")
            except Exception as e:
                logger.warning(f"    Could not verify model device: {e}")
        
        start_time = time.time()

        try:
            # Run Cellpose inference using the model manager
            new_model_path, train_losses, test_losses = train.train_seg(
                self.model_manager.model.net,
                train_data=self.benchmark_data.train_data,
                train_labels=self.benchmark_data.train_labels,
                batch_size=int(self.benchmark_params.get('batch_size', 8)),
                n_epochs=int(self.benchmark_params.get('n_epochs', 1)),
                learning_rate=float(self.benchmark_params.get('learning_rate', 1e-3)),
                weight_decay=float(self.benchmark_params.get('weight_decay', 0.0)),
                nimg_per_epoch=max(2, len(self.benchmark_data.train_data)), # can change this
                model_name="cellpose_benchmark_training"
            )
            
            # Ensure computation is complete (for GPU)
            if 'cuda' in device_context and self.torch_available:
                self.torch.cuda.synchronize()
            elif 'xpu' in device_context and hasattr(self.torch, 'xpu'):
                self.torch.xpu.synchronize()
            
            train_time = time.time() - start_time
            
            # Create model for evaluation using the same device settings
            # Determine GPU setting based on device_context
            use_gpu = device_context != 'cpu'
            
            model = models.CellposeModel(
                gpu=use_gpu,
                pretrained_model=new_model_path
            )
            
            # Move model to the correct device (same as training)
            if use_gpu and hasattr(model, 'net') and model.net is not None:
                try:
                    if device_context.startswith('cuda'):
                        model.net = model.net.to(device_context)
                        logger.debug(f"  Evaluation model moved to {device_context}")
                    elif device_context.startswith('xpu'):
                        model.net = model.net.to('xpu')
                        logger.debug(f"  Evaluation model moved to Intel XPU")
                    elif device_context.startswith('hip'):
                        # AMD GPU using HIP, but PyTorch uses CUDA API
                        model.net = model.net.to(device_context.replace('hip', 'cuda'))
                        logger.debug(f"  Evaluation model moved to AMD GPU via CUDA API")
                except Exception as e:
                    logger.warning(f"  Could not move evaluation model to {device_context}: {e}")

            start_time = time.time()
            
            # run model on test images
            logger.debug(f"    Running evaluation on test images with device: {device_context}")
            masks = model.eval(self.benchmark_data.test_data, batch_size=32)[0]
            
            # Log which device was actually used if we can determine it
            if use_gpu and hasattr(model, 'net') and hasattr(model.net, 'device'):
                actual_device = str(model.net.device) if hasattr(model.net, 'device') else 'unknown'
                logger.debug(f"    Evaluation completed. Model device: {actual_device}")
            
            end_time = time.time() - start_time

            return [train_time, end_time], masks, new_model_path

        except Exception as e:
            # Still return time even if inference failed
            benchmark_time = time.time() - start_time
            logger.debug(f"Inference failed but took {benchmark_time:.4f}s: {e}")
            raise e

    def _check_models_identical(self, model_paths: List[str]) -> bool:
        """
        Check if all trained models are identical by comparing file contents.
        
        This function performs a binary comparison of all model files to
        determine if the training process produces deterministic results.
        
        Args:
            model_paths: List of paths to model files to compare
            
        Returns:
            True if all models are identical, False otherwise
        """
        import filecmp
        
        if len(model_paths) < 2:
            logger.debug("Less than 2 models to compare, skipping check")
            return True
        
        # Use the first model as reference
        reference_model = model_paths[0]
        
        # Check if reference exists
        if not Path(reference_model).exists():
            logger.warning(f"Reference model not found: {reference_model}")
            return False
        
        # Compare all other models against the reference
        all_identical = True
        for i, model_path in enumerate(model_paths[1:], start=2):
            if not Path(model_path).exists():
                logger.warning(f"Model {i} not found: {model_path}")
                all_identical = False
                continue
            
            # Binary file comparison
            if not filecmp.cmp(reference_model, model_path, shallow=False):
                logger.debug(f"Model {i} differs from reference model")
                all_identical = False
                # Continue checking to log all differences
        
        return all_identical
    
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
        This method extracts relevant metrics from all benchmark results (current + historical)
        and structures them in a way that PerformancePlotter can use.
        """
        if not hasattr(self, 'all_results') or not self.all_results:
            logger.warning("No all_results available for plotting")
            return []
        
        plotting_data = []

        # Here we convert all the results we have ingested into a single dictionary which is
        # passed to the PerformancePlotter and will be used as the data for plotting.
        # The idea is to not have any unnecessary data in the dictionary, only what is needed for plotting.
        
        # Iterate through all loaded results (self.all_results is a list of result dictionaries)
        for result_dict in self.all_results:
            # Extract device results from this specific result
            device_results = result_dict.get('results', {}).get('device_results', {})
            
            # Get flavour from this result's metadata
            flavour = result_dict.get('flavour', 'unknown')
            
            for device_key, device_data in device_results.items():
                # Parse device name and type from the key (e.g., "CPU_cpu", "Intel(R) Arc(TM) A770 Graphics_xpu")
                device_parts = device_key.split('_')
                device_type = device_parts[-1]  # Last part is always the type
                device_name = '_'.join(device_parts[:-1])  # Everything else is the name
                
                # Process each test configuration for this device
                for test_key, test_data in device_data.items():
                    if not test_data.get('success', False):
                        continue  # Skip failed tests
                    
                    # Determine vendor from device name, type, and device model
                    device_model = test_data.get('device_model', device_name)
                    vendor = self._extract_vendor_from_device(device_name, device_type, device_model)
                    
                    # Training benchmark doesn't have resolution/precision format
                    # Use default values for training benchmark
                    resolution_str = "training"
                    precision = "mixed"
                    
                    # Get device flavour from test data
                    device_flavour = test_data.get('arkitekt_flavour', 'unknown')
                    
                    # Create flavour name for legend
                    flavour_name = self._create_flavour_name(device_flavour,
                                                             device_type)
                    
                    # Calculate metrics from processed_metrics
                    processed_metrics = test_data.get('processed_metrics', {})
                    # Use mean_total_time for performance, fallback to sum
                    train_time = processed_metrics.get('mean_training_time', 0.0)
                    infer_time = processed_metrics.get('mean_inference_time', 0.0)
                    mean_time = processed_metrics.get('mean_total_time',
                                                      train_time + infer_time)
                    image_size_pixels = 1  # Not applicable for training
                    # Iterations per second
                    throughput = 1 / mean_time if mean_time > 0 else 0
                    
                    # Create data point for PerformancePlotter
                    data_point = {
                        'device_name': device_name,
                        'device_type': device_type,
                        'device_model': device_model,
                        'flavour_name': flavour_name,
                        'vendor': vendor,
                        'flavour': flavour,
                        'execution_time': mean_time,
                        'benchmark_title': 'Cellpose Training Performance',
                        
                        # Complex benchmark data for inference time vs image size plots
                        'image_size_pixels': image_size_pixels,
                        'inference_time': mean_time,
                        'throughput': throughput,
                        'precision': precision,
                        'resolution': resolution_str,
                        
                        # Additional metadata
                        'warmup_iterations': test_data.get('warmup_iterations', 0),
                        'benchmark_iterations': test_data.get('benchmark_iterations', 0),
                        'successful_iterations': test_data.get('successful_iterations', 0),
                        'failed_iterations': test_data.get('failed_iterations', 0),
                        
                        # Performance metrics for detailed analysis
                        'performance_metrics': {
                            test_key: {
                                'success': True,
                                'resolution': resolution_str,
                                'image_size_pixels': image_size_pixels,
                                'inference_time': mean_time,
                                'throughput_px_per_sec': throughput,
                                'precision_used': precision,
                                'num_cells': 0,  # Cellpose would normally detect cells
                                'warmup_iterations': test_data.get('warmup_iterations', 0),
                                'benchmark_iterations': test_data.get('benchmark_iterations', 0),
                                'successful_iterations': test_data.get('successful_iterations', 0),
                                'failed_iterations': test_data.get('failed_iterations', 0),
                                **processed_metrics  # Include all the statistical metrics
                            }
                        },
                        
                        # Results summary
                        'results': {
                            'tests_completed': 1,
                            'tests_failed': 0 if test_data.get('success', False) else 1
                        }
                    }
                    
                    plotting_data.append(data_point)
        
        logger.debug(f"Converted {len(plotting_data)} benchmark results for plotting")
        logger.debug(f"Sample plotting data point: {plotting_data[0] if plotting_data else 'None'}")
        
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
            test_idx: Index of the test image (0, 1, or 2)
            
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
        
        For training benchmark:
        1. Loads all inference masks from benchmark results
        2. Creates ground truth comparison plots for each test image
        3. Creates device comparison matrix
        4. Saves evaluation plots alongside performance plots
        """
        logger.info("Starting mask evaluation and comparison...")
        
        if not hasattr(self, 'all_results') or not self.all_results:
            logger.warning("No results available for evaluation")
            return
        
        # Group masks by configuration
        mask_groups = self._group_masks_by_config()
        
        if not mask_groups:
            logger.warning("No masks found for evaluation")
            return

        # Create evaluation plots for each configuration
        for config_key, mask_data in mask_groups.items():
            logger.info(f"Creating evaluation plots for {config_key}")
            
            # Create ground truth comparisons for training benchmark
            self._create_training_ground_truth_comparison(config_key, mask_data)
            self._create_training_ground_truth_comparison_reduced_cpu(config_key, mask_data)
            
            # Create device comparison matrix
            self._create_device_comparison_matrix(config_key, mask_data)
            self._create_device_comparison_matrix_reduced_cpu(config_key,
                                                              mask_data)
        
        # Create model comparison visualizations
        logger.info("Starting model comparison and analysis...")
        self._create_model_comparison_analysis()

    def _check_ground_truth_data_exists(self, config_key: str) -> bool:
        """
        Check if ground truth and original image data exists for a given configuration.
        
        Args:
            config_key: Configuration key like "512x512_float32"
            
        Returns:
            True if both ground truth and original image files exist, False otherwise
        """
        try:
            # Parse resolution from config_key (e.g., "512x512_float32" -> (512, 512))
            resolution_str = config_key.split('_')[0]  # Get "512x512" part
            width, height = map(int, resolution_str.split('x'))
            
            # Check for ground truth and original image files
            dataset = self.config.get('dataset', 'default_dataset')
            data_dir = Path.cwd() / dataset / self.name / "data"
            gt_file = data_dir / f"ground_truth_{width}x{height}.npy"
            img_file = data_dir / f"artificial_image_{width}x{height}.npy"
            
            return gt_file.exists() and img_file.exists()
            
        except Exception as e:
            logger.warning(f"Error checking ground truth data for {config_key}: {e}")
            return False

    def _group_masks_by_config(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group masks by test configuration and test image index for comparison.
        
        For training benchmark, there's one configuration per device,
        but 3 test images per configuration.
        
        Returns:
            Dict with keys like "cellpose_training_test0", "cellpose_training_test1", etc.
            and values as list of mask data for that test image
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
                    
                    # Process each test image mask separately
                    mask_filenames = test_data.get('mask_filenames', [])
                    for test_idx, mask_filename in enumerate(mask_filenames):
                        # Create mask data entry for this test image
                        mask_entry = {
                            'device_name': test_data.get('device_name', 'Unknown'),
                            'device_type': test_data.get('device_type', 'unknown'),
                            'device_model': test_data.get('device_model',
                                                          'Unknown'),
                            'flavour': flavour,
                            'timestamp': timestamp,
                            'mask_filename': mask_filename,
                            'test_idx': test_idx,
                            # Training benchmark doesn't have resolution/precision
                            'resolution': None,
                            'precision': None,
                            'inference_time': (
                                test_data.get('processed_metrics', {})
                                .get('mean_inference_time', 0.0))
                        }
                        
                        # Group by test configuration + test image index
                        config_key = f"{test_key}test{test_idx}"
                        if config_key not in mask_groups:
                            mask_groups[config_key] = []
                        mask_groups[config_key].append(mask_entry)
        
        logger.debug(f"Grouped masks into {len(mask_groups)} "
                     f"configurations: {list(mask_groups.keys())}")
        return mask_groups
    
    def _create_training_ground_truth_comparison(self, config_key: str, mask_data: List[Dict[str, Any]]):
        """
        Create ground truth comparison visualization for training benchmark.
        
        Loads test images and ground truth from the training data directory,
        then compares device predictions against ground truth.
        """
        if not mask_data:
            return
        
        try:
            # Extract test index from config_key (e.g., "cellpose_training test0" -> 0)
            test_idx_str = config_key.split('test')[-1]
            test_idx = int(test_idx_str) if test_idx_str.isdigit() else 0
            
            # Load test data and ground truth from training data
            from cellpose import io
            dataset = self.config.get('dataset', 'default_dataset')
            data_dir = Path.cwd() / dataset / self.name / "data"
            test_dir = data_dir / "human_in_the_loop" / "test"
            
            masks_ext = "_seg.npy"
            test_data_loaded, test_labels_loaded, _ = io.load_images_labels(
                str(test_dir), mask_filter=masks_ext)
            
            if test_idx >= len(test_data_loaded):
                logger.warning(f"Test index {test_idx} out of range")
                return
            
            original_image = test_data_loaded[test_idx]
            gt_mask = test_labels_loaded[test_idx]
            
            logger.info(f"  Using test image {test_idx} and ground truth from training data")
            
            # Create comparison plot
            import matplotlib.pyplot as plt
            
            n_devices = len(mask_data)
            fig, axes = plt.subplots(2, n_devices + 1, figsize=(4*(n_devices + 1), 8))
            
            # Top-left: Original image
            if len(original_image.shape) == 3:
                # Handle multi-channel images (e.g., 2 channels for cytoplasm + nucleus)
                if original_image.shape[0] in [1, 2, 3, 4]:
                    # Channels first format
                    if original_image.shape[0] == 1:
                        axes[0, 0].imshow(original_image[0], cmap='gray')
                    elif original_image.shape[0] == 2:
                        axes[0, 0].imshow(original_image[0], cmap='gray')
                    elif original_image.shape[0] == 3:
                        axes[0, 0].imshow(np.transpose(original_image, (1, 2, 0)))
                    else:
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
            
            plt.suptitle(f'Cellpose Training Ground Truth Comparison - Test Image {test_idx}\n'
                        f'Top: Instance Masks | Bottom: Binary Mask Differences (White=Match, Red=Differ)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            self._save_evaluation_plot(fig, f'ground_truth_comparison_{config_key}')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create training ground truth comparison for {config_key}: {e}")
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
            
            # Load test data and ground truth
            from cellpose import io
            dataset = self.config.get('dataset', 'default_dataset')
            data_dir = Path.cwd() / dataset / self.name / "data"
            test_dir = data_dir / "human_in_the_loop" / "test"
            
            masks_ext = "_seg.npy"
            test_data_loaded, test_labels_loaded, _ = io.load_images_labels(
                str(test_dir), mask_filter=masks_ext)
            
            if test_idx >= len(test_data_loaded):
                logger.warning(f"Test index {test_idx} out of range")
                return
            
            original_image = test_data_loaded[test_idx]
            gt_mask = test_labels_loaded[test_idx]
            
            # Create comparison plot
            import matplotlib.pyplot as plt
            
            n_devices = len(filtered_data)
            fig, axes = plt.subplots(2, n_devices + 1, figsize=(4*(n_devices + 1), 8))
            
            # Top-left: Original image
            if len(original_image.shape) == 3:
                if original_image.shape[0] in [1, 2, 3, 4]:
                    if original_image.shape[0] == 1:
                        axes[0, 0].imshow(original_image[0], cmap='gray')
                    elif original_image.shape[0] == 2:
                        axes[0, 0].imshow(original_image[0], cmap='gray')
                    elif original_image.shape[0] == 3:
                        axes[0, 0].imshow(np.transpose(original_image, (1, 2, 0)))
                    else:
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
            
            plt.suptitle(f'Cellpose Training Ground Truth Comparison (Reduced CPU) - Test Image {test_idx}\n'
                        f'Top: Instance Masks | Bottom: Binary Mask Differences', 
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
        
        Shows differences between all pairs of devices.
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
                    shape_str = (mask.shape if hasattr(mask, 'shape')
                                else 'N/A')
                    logger.debug(f"Loaded mask type: {type(mask)}, "
                                 f"shape: {shape_str}")
                    # Normalize mask for consistent visualization
                    normalized_mask = self._normalize_instance_mask(mask)
                    logger.debug(f"Adding to loaded_masks: "
                                 f"({type(mask)}, {type(normalized_mask)})")
                    loaded_masks.append((mask, normalized_mask))
                    
                    # Get device flavour information for consistent naming
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
                        # Diagonal: show normalized original mask with dynamic colormap
                        mask_max_instances = np.max(normalized_mask_i) if np.max(normalized_mask_i) > 0 else 1
                        axes[i][j].imshow(normalized_mask_i, cmap='viridis', vmin=0, vmax=mask_max_instances)
                        axes[i][j].set_title(device_info[i]['title'], fontsize=10, fontweight='bold')
                    elif i > j:
                        # Lower triangle: show binary difference visualization
                        # Convert instance masks to binary masks for meaningful comparison
                        binary_i = (original_mask_i > 0).astype(float)
                        binary_j = (original_mask_j > 0).astype(float)
                        
                        # Create binary difference mask
                        diff_binary = (binary_i != binary_j).astype(float)
                        
                        # Create RGB image: white background, red differences
                        diff_rgb = np.ones((diff_binary.shape[0], diff_binary.shape[1], 3))
                        diff_rgb[:, :, 1] = 1 - diff_binary  # Green channel
                        diff_rgb[:, :, 2] = 1 - diff_binary  # Blue channel
                        
                        axes[i][j].imshow(diff_rgb)
                        
                        # Calculate metrics
                        diff_pixels = int(diff_binary.sum())
                        total_pixels = int(diff_binary.size)
                        diff_percent = (diff_pixels / total_pixels) * 100
                        
                        # Format percentage with scientific notation if very small
                        if diff_percent < 0.01 and diff_percent > 0:
                            diff_percent_str = f"{diff_percent:.2e}%"
                        else:
                            diff_percent_str = f"{diff_percent:.2f}%"
                        
                        cells_i = len(np.unique(original_mask_i)) - 1
                        cells_j = len(np.unique(original_mask_j)) - 1
                        
                        axes[i][j].set_title(f"Binary Mask Differences\n({diff_percent_str}, {diff_pixels:,} pixels)\n{cells_i} vs {cells_j} cells", 
                                           fontsize=9, fontweight='bold')
                        
                        # Add black border around difference images
                        for spine in axes[i][j].spines.values():
                            spine.set_visible(True)
                            spine.set_color('black')
                            spine.set_linewidth(3)
                        axes[i][j].tick_params(which='both', length=0)
                    else:
                        # Upper triangle: hide these plots (redundant)
                        axes[i][j].set_visible(False)
                    
                    # Set axis properties
                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])
            
            # Create title for training benchmark with test image index
            # Extract test index from config_key (e.g., "cellpose_training test0" -> "0")
            test_idx = config_key.split('test')[-1] if 'test' in config_key else '0'
            plt.suptitle(f'Cellpose Training Device-to-Device '
                        f'Comparison Matrix - Test Image {test_idx}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_name = f'device_comparison_matrix_{config_key}'
            self._save_evaluation_plot(fig, plot_name)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create device comparison matrix for {config_key}: {e}")

    def _create_device_comparison_matrix_reduced_cpu(self, config_key: str, mask_data: List[Dict[str, Any]]):
        """
        Create device-to-device comparison matrix with reduced CPU representation.
        
        If multiple CPU devices are present, only keeps one CPU device to avoid redundancy.
        Shows differences between all pairs of devices.
        """
        if len(mask_data) < 2:
            logger.debug(f"Not enough devices for reduced CPU comparison matrix in {config_key}")
            return
        
        try:
            # Filter mask data to reduce CPU devices to one representative
            filtered_mask_data = self._reduce_cpu_devices(mask_data)
            
            # Check if we actually need the reduced CPU version
            # If the filtering didn't change anything, skip creating duplicate
            if len(filtered_mask_data) == len(mask_data):
                logger.debug(f"No CPU reduction needed for {config_key} - skipping reduced CPU comparison matrix")
                return
            
            if len(filtered_mask_data) < 2:
                logger.debug(f"Not enough devices after CPU reduction for comparison matrix in {config_key}")
                return
            
            # Load all masks and normalize them
            loaded_masks = []
            device_info = []
            
            for mask_entry in filtered_mask_data:
                mask = self._load_mask(mask_entry)
                if mask is not None:
                    # Normalize mask for consistent visualization
                    normalized_mask = self._normalize_instance_mask(mask)
                    loaded_masks.append((mask, normalized_mask))
                    
                    # Get device flavour information for consistent naming
                    device_flavour = mask_entry.get('flavour', 'unknown')
                    device_type = mask_entry['device_type']
                    device_model = mask_entry.get('device_model', mask_entry['device_name'])
                    
                    # Create title: remove flavour only for CPU devices in reduced CPU version
                    if device_type.lower() == 'cpu':
                        title = device_model
                        flavour_name = ''
                    else:
                        flavour_name = self._create_flavour_name(device_flavour, device_type)
                        title = f"{flavour_name}:\n{device_model}"
                    
                    device_info.append({
                        'flavour_name': flavour_name,
                        'device_model': device_model,
                        'title': title
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
                        # Diagonal: show normalized original mask with dynamic colormap
                        mask_max_instances = np.max(normalized_mask_i) if np.max(normalized_mask_i) > 0 else 1
                        axes[i][j].imshow(normalized_mask_i, cmap='viridis', vmin=0, vmax=mask_max_instances)
                        axes[i][j].set_title(device_info[i]['title'], fontsize=10, fontweight='bold')
                    elif i > j:
                        # Lower triangle: show binary difference visualization
                        # Convert instance masks to binary masks for meaningful comparison
                        binary_i = (original_mask_i > 0).astype(float)
                        binary_j = (original_mask_j > 0).astype(float)
                        
                        # Create binary difference mask
                        diff_binary = (binary_i != binary_j).astype(float)
                        
                        # Create RGB image: white background, red differences
                        diff_rgb = np.ones((diff_binary.shape[0], diff_binary.shape[1], 3))
                        diff_rgb[:, :, 1] = 1 - diff_binary  # Green channel
                        diff_rgb[:, :, 2] = 1 - diff_binary  # Blue channel
                        
                        axes[i][j].imshow(diff_rgb)
                        
                        # Calculate metrics
                        diff_pixels = int(diff_binary.sum())
                        total_pixels = int(diff_binary.size)
                        diff_percent = (diff_pixels / total_pixels) * 100
                        
                        # Format percentage with scientific notation if very small
                        if diff_percent < 0.01 and diff_percent > 0:
                            diff_percent_str = f"{diff_percent:.2e}%"
                        else:
                            diff_percent_str = f"{diff_percent:.2f}%"
                        
                        cells_i = len(np.unique(original_mask_i)) - 1
                        cells_j = len(np.unique(original_mask_j)) - 1
                        
                        axes[i][j].set_title(f"Binary Mask Differences\n({diff_percent_str}, {diff_pixels:,} pixels)\n{cells_i} vs {cells_j} cells", 
                                           fontsize=9, fontweight='bold')
                        
                        # Add black border around difference images
                        for spine in axes[i][j].spines.values():
                            spine.set_visible(True)
                            spine.set_color('black')
                            spine.set_linewidth(3)
                        axes[i][j].tick_params(which='both', length=0)
                    else:
                        # Upper triangle: hide these plots (redundant)
                        axes[i][j].set_visible(False)
                    
                    # Set axis properties
                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])
            
            # Create title for training benchmark with test image index
            # Extract test index from config_key
            test_idx = config_key.split('test')[-1] if 'test' in config_key else '0'
            plt.suptitle(f'Cellpose Training Device Comparison (Reduced CPU) '
                         f'- Test Image {test_idx}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_name = f'device_comparison_matrix_reduced_cpu_{config_key}'
            self._save_evaluation_plot(fig, plot_name)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create reduced CPU device comparison "
                         f"matrix for {config_key}: {e}")

    def _reduce_cpu_devices(self, mask_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reduce CPU devices to only one representative to avoid redundancy.
        
        Keeps all non-CPU devices and only the first CPU device found.
        
        Args:
            mask_data: List of mask entries with device information
            
        Returns:
            Filtered list with at most one CPU device
        """
        filtered_data = []
        cpu_found = False
        
        for mask_entry in mask_data:
            device_type = mask_entry.get('device_type', '').lower()
            
            if device_type == 'cpu':
                if not cpu_found:
                    filtered_data.append(mask_entry)
                    cpu_found = True
                # Skip additional CPU devices
            else:
                # Keep all non-CPU devices
                filtered_data.append(mask_entry)
        
        return filtered_data
    
    def _count_non_overlapped_cells(self, mask: np.ndarray) -> int:
        """
        Count cells in instance segmentation mask, excluding those that are 100% overlapped by another cell.
        
        A cell is considered 100% overlapped if all of its pixels are also covered by another cell instance.
        This helps exclude segmentation artifacts or completely obscured cells.
        
        Args:
            mask: Instance segmentation mask with unique integer IDs for each cell
            
        Returns:
            Number of non-overlapped cells
        """
        if mask is None:
            return 0
        
        # Get unique instance IDs (excluding background)
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
        
        if len(unique_ids) == 0:
            return 0
        
        non_overlapped_count = 0
        
        for cell_id in unique_ids:
            # Get pixels belonging to this cell
            cell_pixels = (mask == cell_id)
            
            # Check if this cell is 100% overlapped by any other cell
            is_fully_overlapped = False
            
            for other_id in unique_ids:
                if other_id == cell_id:
                    continue
                
                # Get pixels belonging to the other cell
                other_pixels = (mask == other_id)
                
                # Check if all pixels of current cell are also covered by other cell
                # This happens when the current cell is completely contained within another
                if np.all(cell_pixels <= other_pixels):
                    is_fully_overlapped = True
                    break
            
            # Count cell only if it's not fully overlapped
            if not is_fully_overlapped:
                non_overlapped_count += 1
        
        return non_overlapped_count

    def _normalize_instance_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Normalize instance segmentation mask for consistent visualization.
        
        Relabels instances based on their spatial position (centroid) to ensure
        consistent coloring across different masks when displayed with the same colormap.
        
        Args:
            mask: Instance segmentation mask with unique integer IDs for each cell
            
        Returns:
            Normalized mask with spatially-consistent instance labeling
        """
        if mask is None:
            return mask
        
        # Create output mask (background stays 0)
        normalized_mask = np.zeros_like(mask)
        
        # Get unique instance IDs (excluding background)
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
        
        if len(unique_ids) == 0:
            return normalized_mask
        
        # Calculate centroids for each instance
        centroids = []
        for instance_id in unique_ids:
            y_coords, x_coords = np.where(mask == instance_id)
            if len(y_coords) > 0:
                centroid_y = np.mean(y_coords)
                centroid_x = np.mean(x_coords)
                centroids.append((centroid_y, centroid_x, instance_id))
        
        # Sort instances by spatial position (top-left to bottom-right)
        # Primary sort: y-coordinate (top to bottom)
        # Secondary sort: x-coordinate (left to right)
        centroids.sort(key=lambda c: (c[0], c[1]))
        
        # Relabel instances with consecutive IDs starting from 1
        for new_id, (_, _, original_id) in enumerate(centroids, start=1):
            normalized_mask[mask == original_id] = new_id
        
        return normalized_mask

    def _load_mask(self, mask_entry: Dict[str, Any]) -> np.ndarray:
        """Load mask from file based on mask entry."""
        try:
            flavour = mask_entry['flavour']
            timestamp = mask_entry['timestamp']  # Use timestamp from mask_entry, not self.timestamp
            mask_filename = mask_entry['mask_filename']
            
            # Construct path to mask file using the correct timestamp
            mask_dir = Path(self.config.get('dataset', 'default_dataset')) / self.name / f"{timestamp}_{flavour}"
            mask_path = mask_dir / mask_filename
            
            if mask_path.exists():
                loaded_data = np.load(mask_path, allow_pickle=True)
                shape_str = (loaded_data.shape if hasattr(loaded_data, 'shape')
                            else 'no shape')
                logger.debug(f"Loaded mask shape: {shape_str}, "
                             f"type: {type(loaded_data)}")
                
                # Handle case where mask might be saved as object array
                # containing a list
                if loaded_data.dtype == object:
                    # This might be a 0-d array containing a list
                    if loaded_data.shape == ():
                        actual_mask = loaded_data.item()
                        if (isinstance(actual_mask, list) and
                                len(actual_mask) > 0):
                            logger.debug(f"Extracted first mask from list "
                                         f"of {len(actual_mask)} masks")
                            return actual_mask[0]
                        else:
                            return actual_mask
                    else:
                        logger.warning(f"Unexpected object array shape: "
                                       f"{loaded_data.shape}")
                        return loaded_data
                else:
                    return loaded_data
            else:
                logger.warning(f"Mask file not found: {mask_path}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load mask: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _create_model_comparison_analysis(self):
        """
        Create comprehensive model comparison analysis.
        
        Compares trained models from different devices/flavours using:
        1. Parameter statistics (mean, std, min, max)
        2. Weight distribution comparisons
        3. Layer-wise difference metrics
        4. Similarity heatmap matrix
        """
        logger.info("Creating model comparison analysis...")
        
        # Collect all model paths from results
        model_info = self._collect_model_info()
        
        if len(model_info) < 2:
            logger.warning("Not enough models for comparison (need at least 2)")
            return
        
        # Create comparison matrix and visualizations
        self._create_model_comparison_matrix(model_info)
        self._create_model_statistics_table(model_info)
        self._create_weight_distribution_comparison(model_info)
    
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
                    
                    # Use the first model from iterations (they should be identical)
                    model_path = model_paths[0]
                    
                    # Get device flavour information for consistent naming
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
        Create a similarity matrix comparing all models.
        
        Uses multiple metrics:
        - Parameter-wise L2 distance
        - Cosine similarity
        - Max absolute difference
        """
        import matplotlib.pyplot as plt
        import torch
        
        n_models = len(model_info)
        if n_models < 2:
            return
        
        logger.info(f"Creating model comparison matrix for {n_models} models...")
        
        # Load all models
        models = []
        labels = []
        for info in model_info:
            try:
                model = torch.load(info['model_path'], map_location='cpu')
                models.append(model)
                labels.append(info['title'])
            except Exception as e:
                logger.warning(f"Failed to load model {info['model_path']}: {e}")
                return
        
        # Create comparison matrices for different metrics
        l2_distance_matrix = np.zeros((n_models, n_models))
        cosine_sim_matrix = np.zeros((n_models, n_models))
        max_diff_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    l2_distance_matrix[i, j] = 0
                    cosine_sim_matrix[i, j] = 1.0
                    max_diff_matrix[i, j] = 0
                else:
                    metrics = self._compare_two_models(models[i], models[j])
                    l2_distance_matrix[i, j] = metrics['l2_distance']
                    cosine_sim_matrix[i, j] = metrics['cosine_similarity']
                    max_diff_matrix[i, j] = metrics['max_abs_diff']
        
        # Create visualization with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: L2 Distance (lower is more similar)
        im1 = axes[0].imshow(l2_distance_matrix, cmap='YlOrRd', aspect='auto')
        axes[0].set_title('L2 Distance\n(Lower = More Similar)', fontsize=12, fontweight='bold')
        axes[0].set_xticks(range(n_models))
        axes[0].set_yticks(range(n_models))
        axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[0].set_yticklabels(labels, fontsize=8)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                text = axes[0].text(j, i, f'{l2_distance_matrix[i, j]:.2e}',
                                   ha="center", va="center", color="black", fontsize=7)
        
        # Plot 2: Cosine Similarity (higher is more similar)
        im2 = axes[1].imshow(cosine_sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[1].set_title('Cosine Similarity\n(Higher = More Similar)', fontsize=12, fontweight='bold')
        axes[1].set_xticks(range(n_models))
        axes[1].set_yticks(range(n_models))
        axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[1].set_yticklabels(labels, fontsize=8)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                text = axes[1].text(j, i, f'{cosine_sim_matrix[i, j]:.6f}',
                                   ha="center", va="center", color="black", fontsize=7)
        
        # Plot 3: Max Absolute Difference (lower is more similar)
        im3 = axes[2].imshow(max_diff_matrix, cmap='YlOrRd', aspect='auto')
        axes[2].set_title('Max Absolute Difference\n(Lower = More Similar)', fontsize=12, fontweight='bold')
        axes[2].set_xticks(range(n_models))
        axes[2].set_yticks(range(n_models))
        axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[2].set_yticklabels(labels, fontsize=8)
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                text = axes[2].text(j, i, f'{max_diff_matrix[i, j]:.2e}',
                                   ha="center", va="center", color="black", fontsize=7)
        
        plt.suptitle('Model Weight Similarity Comparison Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self._save_evaluation_plot(fig, 'model_comparison_matrix')
        plt.close(fig)
    
    def _compare_two_models(self, model1: dict, model2: dict) -> Dict[str, float]:
        """
        Compare two model state dictionaries using multiple metrics.
        
        Returns:
            Dictionary with comparison metrics
        """
        import torch
        
        # Get common keys (parameters that exist in both models)
        common_keys = set(model1.keys()) & set(model2.keys())
        
        if not common_keys:
            return {
                'l2_distance': float('inf'),
                'cosine_similarity': 0.0,
                'max_abs_diff': float('inf')
            }
        
        # Flatten all parameters into single vectors
        params1 = []
        params2 = []
        
        for key in sorted(common_keys):
            if isinstance(model1[key], torch.Tensor) and isinstance(model2[key], torch.Tensor):
                # Only compare tensors (skip non-tensor metadata)
                if model1[key].shape == model2[key].shape:
                    params1.append(model1[key].flatten().float())
                    params2.append(model2[key].flatten().float())
        
        if not params1:
            return {
                'l2_distance': 0.0,
                'cosine_similarity': 1.0,
                'max_abs_diff': 0.0
            }
        
        # Concatenate all parameters
        vec1 = torch.cat(params1)
        vec2 = torch.cat(params2)
        
        # Calculate metrics
        # L2 distance (Euclidean distance)
        l2_dist = torch.norm(vec1 - vec2, p=2).item()
        
        # Cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            vec1.unsqueeze(0), vec2.unsqueeze(0)
        ).item()
        
        # Max absolute difference
        max_diff = torch.max(torch.abs(vec1 - vec2)).item()
        
        return {
            'l2_distance': l2_dist,
            'cosine_similarity': cosine_sim,
            'max_abs_diff': max_diff
        }
    
    def _create_model_statistics_table(self, model_info: List[Dict[str, Any]]):
        """
        Create a detailed statistics table for all models.
        
        Shows per-model statistics like:
        - Number of parameters
        - Mean, std, min, max of weights
        - File size
        """
        import matplotlib.pyplot as plt
        import torch
        
        logger.info("Creating model statistics table...")
        
        # Collect statistics for each model
        stats_data = []
        
        for info in model_info:
            try:
                model = torch.load(info['model_path'], map_location='cpu')
                
                # Calculate statistics
                all_params = []
                n_params = 0
                n_layers = 0
                
                for key, value in model.items():
                    if isinstance(value, torch.Tensor):
                        n_layers += 1
                        n_params += value.numel()
                        all_params.append(value.flatten().float())
                
                if all_params:
                    all_params_tensor = torch.cat(all_params)
                    
                    stats = {
                        'Device': info['title'].replace('\n', ' '),
                        'Layers': n_layers,
                        'Parameters': f"{n_params:,}",
                        'Mean': f"{all_params_tensor.mean().item():.6f}",
                        'Std': f"{all_params_tensor.std().item():.6f}",
                        'Min': f"{all_params_tensor.min().item():.6f}",
                        'Max': f"{all_params_tensor.max().item():.6f}",
                        'File Size': f"{Path(info['model_path']).stat().st_size / 1024 / 1024:.2f} MB"
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
        
        # Prepare table data
        columns = list(stats_data[0].keys())
        rows = [[stats[col] for col in columns] for stats in stats_data]
        
        # Create table
        table = ax.table(cellText=rows, colLabels=columns, cellLoc='center',
                        loc='center', bbox=[0, 0, 1, 1])
        
        # Style the table
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
    
    def _create_weight_distribution_comparison(self, model_info: List[Dict[str, Any]]):
        """
        Create overlaid histograms showing weight distributions for each model.
        
        This helps visualize if models have similar weight distributions
        even if individual values differ.
        """
        import matplotlib.pyplot as plt
        import torch
        
        logger.info("Creating weight distribution comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Collect weight distributions
        for idx, info in enumerate(model_info):
            try:
                model = torch.load(info['model_path'], map_location='cpu')
                
                # Collect all weights
                all_weights = []
                for key, value in model.items():
                    if isinstance(value, torch.Tensor):
                        all_weights.append(value.flatten().float())
                
                if all_weights:
                    weights_tensor = torch.cat(all_weights).numpy()
                    
                    # Plot 1: Full distribution
                    axes[0].hist(weights_tensor, bins=100, alpha=0.5, 
                               label=info['title'].replace('\n', ' '), density=True)
                    
                    # Plot 2: Log scale (better for long tails)
                    axes[1].hist(weights_tensor, bins=100, alpha=0.5, 
                               label=info['title'].replace('\n', ' '), 
                               density=True, log=True)
                    
                    # Plot 3: Zoomed in around zero
                    weights_near_zero = weights_tensor[np.abs(weights_tensor) < 0.1]
                    axes[2].hist(weights_near_zero, bins=100, alpha=0.5,
                               label=info['title'].replace('\n', ' '), density=True)
                    
                    # Plot 4: Box plot summary
                    axes[3].boxplot([weights_tensor], positions=[idx],
                                  labels=[info['title'].replace('\n', ' ')],
                                  showfliers=False)
                    
            except Exception as e:
                logger.warning(f"Failed to plot distribution for {info['model_path']}: {e}")
        
        # Configure plots
        axes[0].set_title('Weight Distribution (Full Range)', fontweight='bold')
        axes[0].set_xlabel('Weight Value')
        axes[0].set_ylabel('Density')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Weight Distribution (Log Scale)', fontweight='bold')
        axes[1].set_xlabel('Weight Value')
        axes[1].set_ylabel('Density (log scale)')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title('Weight Distribution (Near Zero)', fontweight='bold')
        axes[2].set_xlabel('Weight Value')
        axes[2].set_ylabel('Density')
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        
        axes[3].set_title('Weight Distribution Summary (Box Plot)', fontweight='bold')
        axes[3].set_ylabel('Weight Value')
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Model Weight Distribution Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self._save_evaluation_plot(fig, 'model_weight_distributions')
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
        logger.info("Evaluating Cellpose benchmark results with mask comparison...")
        
        # First create performance plots
        from ..core.visualizations import PerformancePlotter
        
        # Convert real benchmark results for plotting
        plotting_data = self._convert_results_for_performance_plot()
        
        try:
            # Create performance plotter with real data
            plotter = PerformancePlotter(plotting_data)
            
            # Create plot path with evaluation timestamp (not benchmark timestamp)
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
