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

class CellposeArtificialInference(PyTorchBenchmark):
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

    def _run_benchmark(self) -> Dict[str, Any]:
        """
        Run Cellpose artificial inference benchmark.
        
        1. Check if artificial data exists in <benchmark_name>/data folder
        2. If not, create it using _create_artificial_data
        3. For each device, resolution, and precision:
           - Run warmup iterations
           - Run benchmark iterations
           - Collect inference times
        4. Return structured results with raw data and processed metrics
        """
        logger.info(f"Starting Cellpose Artificial Inference Benchmark")
        
        # Check if artificial data exists, create if needed
        data_dir = self._ensure_artificial_data_exists()
        
        # Get benchmark parameters from config
        benchmark_params = self.config.get('benchmarks', {}).get('parameters', {}).get(self.name, {})
        warmup_iterations = benchmark_params.get('warmup_iterations', 1)
        num_iterations = benchmark_params.get('num_iterations', 1)
        cpu_resolutions = benchmark_params.get('cpu_resolutions', [[256, 256],[512, 512]])
        gpu_resolutions = benchmark_params.get('gpu_resolutions', [[256, 256],[512, 512]])
        precisions = benchmark_params.get('precisions', ['float32'])
        
        import cellpose
        logger.debug(f"Cellpose path: {cellpose.__file__}")
        logger.debug(f"Benchmark parameters:")
        logger.debug(f"  Warmup iterations: {warmup_iterations}")
        logger.debug(f"  Benchmark iterations: {num_iterations}")
        logger.debug(f"  CPU resolutions: {cpu_resolutions}")
        logger.debug(f"  GPU resolutions: {gpu_resolutions}")
        logger.debug(f"  Precisions: {precisions}")
        
        # Run benchmarks for each device
        all_results = {}
        devices = self.device_manager._list_available_devices()
        
        for device in devices:
            logger.info(f"\n--- Testing Device: {device.name} ({device.type}) ---")
            
            # Determine which resolutions to use based on device type
            if device.type.lower() == 'cpu':
                resolutions = cpu_resolutions
            else:
                resolutions = gpu_resolutions
            
            device_results = {}
            
            # Setup model for this device
            device_context = self._setup_device_context(device)
            self.model_manager = self._load_model_for_device(self.model_manager, device_context)
            
            for resolution in resolutions:
                height, width = resolution
                logger.info(f"  Testing resolution: {width}x{height}")
                
                # Load artificial data for this resolution
                image_data = self._load_artificial_data(data_dir, resolution)
                
                for precision in precisions:
                    logger.info(f"    Testing precision: {precision}")
                    
                    # Convert image to target precision
                    test_image = self._convert_image_precision(image_data, precision)
                    
                    # Run warmup iterations
                    logger.debug(f"      Running {warmup_iterations} warmup iterations...")
                    for _ in range(warmup_iterations):
                        try:
                            _, _ = self._run_single_inference(self.model_manager, test_image, device_context)
                        except Exception as e:
                            logger.warning(f"Warmup iteration failed: {e}")
                    
                    # Run benchmark iterations and collect times and masks
                    inference_times = []
                    masks_collected = []
                    successful_iterations = 0
                    failed_iterations = 0
                    
                    logger.debug(f"      Running {num_iterations} benchmark iterations...")
                    for iteration in range(num_iterations):
                        try:
                            inference_time, mask = self._run_single_inference(self.model_manager, test_image, device_context)
                            inference_times.append(inference_time)
                            masks_collected.append(mask)
                            successful_iterations += 1
                        except Exception as e:
                            logger.warning(f"Benchmark iteration {iteration+1} failed: {e}")
                            failed_iterations += 1
                    
                    # Store the best mask (from fastest inference) and create mask filename
                    mask_filename = None
                    if masks_collected and inference_times:
                        # Use mask from fastest successful inference
                        best_mask_idx = np.argmin(inference_times)
                        best_mask = masks_collected[best_mask_idx]
                        
                        # Create mask filename and save
                        mask_filename = self._save_mask(best_mask, device, resolution, precision)
                    
                    # Store results for this configuration
                    test_key = f"{width}x{height}_{precision}"
                    device_results[test_key] = {
                        'device_name': device.name,
                        'device_type': device.type,
                        'device_id': device.id,
                        'device_model': getattr(device, 'cpu_model', device.name) if device.type.lower() == 'cpu' else device.name,
                        'arkitekt_flavour': getattr(device, 'arkitekt_flavour', 'unknown'),
                        'resolution': [width, height],
                        'precision': precision,
                        'warmup_iterations': warmup_iterations,
                        'benchmark_iterations': num_iterations,
                        'successful_iterations': successful_iterations,
                        'failed_iterations': failed_iterations,
                        'raw_inference_times': inference_times,
                        'processed_metrics': self._calculate_processed_metrics(inference_times),
                        'image_size_pixels': width * height,
                        'mask_filename': mask_filename,  # Add mask filename to results
                        'success': len(inference_times) > 0
                    }
                    
                    if inference_times:
                        avg_time = sum(inference_times) / len(inference_times)
                        logger.info(f"      ✓ Completed: avg={avg_time:.4f}s, runs={len(inference_times)}/{num_iterations}")
                    else:
                        logger.warning(f"      ✗ All iterations failed")
            
            all_results[f"{device.name}_{device.type}"] = device_results
        
        # Calculate overall benchmark statistics
        overall_stats = self._calculate_overall_statistics(all_results)
        
        final_results = {
            'device_results': all_results,
            'overall_statistics': overall_stats,
            'benchmark_metadata': {
                'total_devices_tested': len(devices),
                'total_configurations': sum(len(device_results) for device_results in all_results.values()),
                'data_directory': str(data_dir),
                'benchmark_parameters': benchmark_params
            }
        }
        
        logger.info(f"\n✓ Benchmark completed successfully")
        logger.info(f"  Devices tested: {len(devices)}")
        logger.info(f"  Total configurations: {final_results['benchmark_metadata']['total_configurations']}")
        
        return final_results
    
    def _ensure_artificial_data_exists(self) -> Path:
        """
        Check if artificial data exists for all required resolutions.
        If any data is missing, create the missing data.
        
        Returns:
            Path to the data directory
        """
        # Get dataset from config
        dataset = self.config.get('dataset', 'default_dataset')
        data_dir = Path.cwd() / dataset / self.name / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all unique resolutions from config
        benchmark_params = self.config.get('benchmarks', {}).get('parameters', {}).get(self.name, {})
        cpu_resolutions = benchmark_params.get('cpu_resolutions', [[512, 512]])
        gpu_resolutions = benchmark_params.get('gpu_resolutions', [[1024, 1024]])
        
        # Combine and deduplicate resolutions
        all_resolutions = []
        for res_list in [cpu_resolutions, gpu_resolutions]:
            for res in res_list:
                if res not in all_resolutions:
                    all_resolutions.append(res)
        
        # Check which resolutions are missing
        missing_resolutions = []
        existing_resolutions = []
        
        for resolution in all_resolutions:
            height, width = resolution
            img_file = data_dir / f"artificial_image_{width}x{height}.npy"
            gt_file = data_dir / f"ground_truth_{width}x{height}.npy"
            
            if img_file.exists() and gt_file.exists():
                existing_resolutions.append(resolution)
            else:
                missing_resolutions.append(resolution)
        
        if existing_resolutions:
            logger.info(f"Found existing artificial data for {len(existing_resolutions)} resolutions: {existing_resolutions}")
        
        if missing_resolutions:
            logger.info(f"Creating missing artificial data for {len(missing_resolutions)} resolutions: {missing_resolutions}")
            self._create_missing_artificial_data(data_dir, missing_resolutions)
        else:
            logger.info(f"All required artificial data already exists in: {data_dir}")
        
        return data_dir
    
    def _create_artificial_data(self, data_dir: Path) -> None:
        """
        Create artificial test images for benchmarking.
        
        Args:
            data_dir: Directory where to store the artificial data
        """
        # Get all unique resolutions from config
        benchmark_params = self.config.get('benchmarks', {}).get('parameters', {}).get(self.name, {})
        cpu_resolutions = benchmark_params.get('cpu_resolutions', [[512, 512]])
        gpu_resolutions = benchmark_params.get('gpu_resolutions', [[1024, 1024]])
        
        # Combine and deduplicate resolutions
        all_resolutions = []
        for res_list in [cpu_resolutions, gpu_resolutions]:
            for res in res_list:
                if res not in all_resolutions:
                    all_resolutions.append(res)
        
        logger.info(f"Creating artificial data for {len(all_resolutions)} resolutions: {all_resolutions}")
        
        # Import image generation functionality from the original benchmark
        from ..core.image_generator import ImageGenerator
        
        for resolution in all_resolutions:
            height, width = resolution
            logger.info(f"  Generating {width}x{height} artificial image...")
            
            # Generate artificial image and ground truth
            image, ground_truth_data = ImageGenerator.generate_artificial_image_adaptive(
                (height, width), 
                f"{self.name}_{width}x{height}"
            )
            
            # Save image data
            image_file = data_dir / f"artificial_image_{width}x{height}.npy"
            np.save(image_file, image)
            
            # Save ground truth data
            gt_file = data_dir / f"ground_truth_{width}x{height}.npy"
            np.save(gt_file, ground_truth_data)
            
            logger.info(f"    Saved: {image_file.name} and {gt_file.name}")

    def _create_missing_artificial_data(self, data_dir: Path, missing_resolutions: List[List[int]]) -> None:
        """
        Create artificial test images for specific missing resolutions.
        
        Args:
            data_dir: Directory where to store the artificial data
            missing_resolutions: List of [height, width] resolutions to create
        """
        # Import image generation functionality from the original benchmark
        from ..core.image_generator import ImageGenerator
        
        for resolution in missing_resolutions:
            height, width = resolution
            logger.info(f"  Generating {width}x{height} artificial image...")
            
            # Generate artificial image and ground truth
            image, ground_truth_data = ImageGenerator.generate_artificial_image_adaptive(
                (height, width), 
                f"{self.name}_{width}x{height}"
            )
            
            # Save image data
            image_file = data_dir / f"artificial_image_{width}x{height}.npy"
            np.save(image_file, image)
            
            # Save ground truth data
            gt_file = data_dir / f"ground_truth_{width}x{height}.npy"
            np.save(gt_file, ground_truth_data)
            
            logger.info(f"    Saved: {image_file.name} and {gt_file.name}")
        
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
    
    def _load_artificial_data(self, data_dir: Path, resolution: List[int]) -> np.ndarray:
        """Load artificial image data for given resolution."""
        height, width = resolution
        image_file = data_dir / f"artificial_image_{width}x{height}.npy"
        
        if not image_file.exists():
            raise FileNotFoundError(f"Artificial data not found: {image_file}")
        
        return np.load(image_file)
    
    def _convert_image_precision(self, image: np.ndarray, precision: str) -> np.ndarray:
        """Convert image to target precision."""
        precision_map = {
            'float16': np.float16,
            'float32': np.float32,
            'float64': np.float64,
            'uint8': np.uint8
        }
        
        if precision not in precision_map:
            logger.warning(f"Unknown precision {precision}, using float32")
            return image.astype(np.float32)
        
        return image.astype(precision_map[precision])
    
    def _run_single_inference(self, model_manager, image: np.ndarray, device_context: str) -> tuple:
        """
        Run a single inference and return the execution time and mask.
        
        Args:
            model_manager: Cellpose model manager
            image: Input image
            device_context: Device context string
            
        Returns:
            Tuple of (inference_time, mask) where:
            - inference_time: float, time in seconds
            - mask: np.ndarray, segmentation mask
        """
        start_time = time.time()
        
        try:
            # Run Cellpose inference using the model manager
            masks, flows, styles = model_manager.eval(
                image,
                diameter=None,
                channels=None,
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )
            
            # Ensure computation is complete (for GPU)
            if 'cuda' in device_context and self.torch_available:
                self.torch.cuda.synchronize()
            elif 'xpu' in device_context and hasattr(self.torch, 'xpu'):
                self.torch.xpu.synchronize()
            
            inference_time = time.time() - start_time
            return inference_time, masks
            
        except Exception as e:
            # Still return time even if inference failed
            inference_time = time.time() - start_time
            logger.debug(f"Inference failed but took {inference_time:.4f}s: {e}")
            raise e
    
    def _calculate_processed_metrics(self, inference_times: List[float]) -> Dict[str, float]:
        """Calculate processed metrics from raw inference times."""
        if not inference_times:
            return {}
        
        inference_times = np.array(inference_times)
        
        return {
            'mean_time': float(np.mean(inference_times)),
            'median_time': float(np.median(inference_times)),
            'std_time': float(np.std(inference_times)),
            'min_time': float(np.min(inference_times)),
            'max_time': float(np.max(inference_times)),
            'q25_time': float(np.percentile(inference_times, 25)),
            'q75_time': float(np.percentile(inference_times, 75)),
            'total_iterations': len(inference_times)
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
                    if test_result['raw_inference_times']:
                        total_time += sum(test_result['raw_inference_times'])
        
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
                    
                    # Extract resolution and precision from test_key (e.g., "512x512_float32")
                    resolution_str, precision = test_key.rsplit('_', 1)
                    width, height = map(int, resolution_str.split('x'))
                    
                    # Get device flavour from test data
                    device_flavour = test_data.get('arkitekt_flavour', 'unknown')
                    
                    # Create flavour name for legend
                    flavour_name = self._create_flavour_name(device_flavour, device_type)
                    
                    # Calculate throughput and get processed metrics
                    processed_metrics = test_data.get('processed_metrics', {})
                    mean_time = processed_metrics.get('mean_time', 0.0)
                    image_size_pixels = test_data.get('image_size_pixels', width * height)
                    throughput = image_size_pixels / mean_time if mean_time > 0 else 0
                    
                    # Create data point for PerformancePlotter
                    data_point = {
                        'device_name': device_name,
                        'device_type': device_type,
                        'device_model': device_model,
                        'flavour_name': flavour_name,
                        'vendor': vendor,
                        'flavour': flavour,
                        'execution_time': mean_time,
                        'benchmark_title': 'Cellpose Artificial Inference Performance Benchmark',
                        
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
    
    def _save_mask(self, mask: np.ndarray, device, resolution: List[int], precision: str) -> str:
        """
        Save mask as .npy file alongside results.
        
        Args:
            mask: Segmentation mask array
            device: Device object with name and type
            resolution: [width, height] 
            precision: Precision string (e.g., 'float32')
            
        Returns:
            str: Relative filename of saved mask
        """
        try:
            # Create mask directory in the same location as results.json
            flavour = getattr(device, 'arkitekt_flavour', 'unknown')
            mask_dir = Path(self.config.get('dataset', 'default_dataset')) / self.name / f"{self.timestamp}_{flavour}"
            mask_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mask filename: device_resolution_precision.npy
            device_safe_name = "".join(c for c in device.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            device_safe_name = device_safe_name.replace(' ', '_')
            width, height = resolution
            mask_filename = f"{device_safe_name}_{width}x{height}_{precision}_mask.npy"
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
        
        This method:
        1. Loads all masks from benchmark results
        2. Creates ground truth comparison plots
        3. Creates device comparison matrix
        4. Saves evaluation plots alongside performance plots
        """
        logger.info("Starting mask evaluation and comparison...")
        
        if not hasattr(self, 'all_results') or not self.all_results:
            logger.warning("No results available for evaluation")
            return
        
        # Group masks by resolution and precision for comparison
        mask_groups = self._group_masks_by_config()
        
        if not mask_groups:
            logger.warning("No masks found for evaluation")
            return

        # Create evaluation plots for each resolution/precision combination
        for config_key, mask_data in mask_groups.items():
            # Check if ground truth data exists for this resolution before proceeding
            if not self._check_ground_truth_data_exists(config_key):
                logger.warning(f"Ground truth data not found for {config_key}, skipping evaluation plots")
                continue
                
            logger.info(f"Creating evaluation plots for {config_key}")
            
            # Create ground truth comparison
            self._create_ground_truth_comparison(config_key, mask_data)
            self._create_ground_truth_comparison_reduced_cpu(config_key, mask_data)
            
            # Create device comparison matrix
            self._create_device_comparison_matrix(config_key, mask_data)
            self._create_device_comparison_matrix_reduced_cpu(config_key, mask_data)

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
        Group masks by resolution and precision for comparison.
        
        Returns:
            Dict with keys like "1024x1024_float32" and values as list of mask data
        """
        mask_groups = {}
        
        for result_dict in self.all_results:
            device_results = result_dict.get('results', {}).get('device_results', {})
            flavour = result_dict.get('flavour', 'unknown')
            timestamp = result_dict.get('timestamp', 'unknown')
            
            for device_key, device_data in device_results.items():
                for test_key, test_data in device_data.items():
                    if not test_data.get('success', False) or not test_data.get('mask_filename'):
                        continue
                    
                    # Create mask data entry
                    mask_entry = {
                        'device_name': test_data.get('device_name', 'Unknown'),
                        'device_type': test_data.get('device_type', 'unknown'),
                        'device_model': test_data.get('device_model', 'Unknown'),
                        'flavour': flavour,
                        'timestamp': timestamp,
                        'mask_filename': test_data.get('mask_filename'),
                        'resolution': test_data.get('resolution', [0, 0]),
                        'precision': test_data.get('precision', 'unknown'),
                        'inference_time': test_data.get('processed_metrics', {}).get('mean_time', 0.0)
                    }
                    
                    # Group by test configuration
                    if test_key not in mask_groups:
                        mask_groups[test_key] = []
                    mask_groups[test_key].append(mask_entry)
        
        logger.debug(f"Grouped masks into {len(mask_groups)} configurations: {list(mask_groups.keys())}")
        return mask_groups
    
    def _create_ground_truth_comparison(self, config_key: str, mask_data: List[Dict[str, Any]]):
        """
        Create ground truth comparison visualization.
        
        Shows ground truth and difference plots for each device.
        """
        if not mask_data:
            return
        
        try:
            # Load the actual ground truth from data directory
            # Parse resolution from config_key (e.g., "512x512_float32" -> (512, 512))
            resolution_str = config_key.split('_')[0]  # Get "512x512" part
            width, height = map(int, resolution_str.split('x'))
            
            # Load ground truth mask from data directory
            dataset = self.config.get('dataset', 'default_dataset')
            data_dir = Path.cwd() / dataset / self.name / "data"
            gt_file = data_dir / f"ground_truth_{width}x{height}.npy"
            
            if not gt_file.exists():
                logger.warning(f"Ground truth file not found: {gt_file}")
                return
            
            gt_data = np.load(gt_file, allow_pickle=True).item()
            if not isinstance(gt_data, dict) or 'ground_truth_mask' not in gt_data:
                logger.warning(f"Invalid ground truth data format in: {gt_file}")
                return
            
            gt_mask = gt_data['ground_truth_mask']
            logger.info(f"  Using actual ground truth from: {gt_file.name} for {config_key}")
            
            # Create comparison plot with ground truth column
            import matplotlib.pyplot as plt
            
            n_devices = len(mask_data)
            # Add 1 extra column for ground truth
            fig, axes = plt.subplots(2, n_devices + 1, figsize=(4*(n_devices + 1), 8))
            if n_devices == 0:
                return
            
            # Load the original artificial image for the top-left position
            data_dir = Path.cwd() / dataset / self.name / "data"
            img_file = data_dir / f"artificial_image_{width}x{height}.npy"
            
            if not img_file.exists():
                logger.warning(f"Original image file not found: {img_file}")
                return
            
            original_image = np.load(img_file)
            logger.info(f"  Using original image from: {img_file.name} for {config_key}")
            
            # Normalize ground truth mask for consistent visualization
            gt_mask_normalized = self._normalize_instance_mask(gt_mask)
            gt_max_instances = np.max(gt_mask_normalized) if np.max(gt_mask_normalized) > 0 else 1
            
            # First column, top row: Original input image
            axes[0, 0].imshow(original_image, cmap='gray')
            axes[0, 0].set_title("Original Image\nInput", fontsize=10, fontweight='bold')
            axes[0, 0].axis('off')
            
            # First column, bottom row: Ground truth mask
            axes[1, 0].imshow(gt_mask_normalized, cmap='viridis', vmin=0, vmax=gt_max_instances)
            axes[1, 0].set_title("Ground Truth\nReference", fontsize=9, fontweight='bold')
            axes[1, 0].axis('off')
            
            for idx, mask_entry in enumerate(mask_data):
                device_mask = self._load_mask(mask_entry)
                if device_mask is None:
                    continue
                
                # Adjust column index to account for ground truth column
                col_idx = idx + 1
                
                # Get device flavour information for consistent naming
                device_flavour = mask_entry.get('flavour', 'unknown')
                device_type = mask_entry['device_type']
                flavour_name = self._create_flavour_name(device_flavour, device_type)
                device_model = mask_entry.get('device_model', mask_entry['device_name'])
                
                # Create consistent title format with line break: "Flavour Name:\nDevice Model"
                device_title = f"{flavour_name}:\n{device_model}"
                
                # Normalize device mask for consistent visualization
                device_mask_normalized = self._normalize_instance_mask(device_mask)
                device_max_instances = np.max(device_mask_normalized) if np.max(device_mask_normalized) > 0 else 1
                
                # Top row: Original masks with improved titles and dynamic colormap range
                axes[0, col_idx].imshow(device_mask_normalized, cmap='viridis', vmin=0, vmax=device_max_instances)
                axes[0, col_idx].set_title(device_title, fontsize=10, fontweight='bold')
                axes[0, col_idx].axis('off')
                
                # Bottom row: Binary difference visualization with black borders
                # Convert instance masks to binary masks for meaningful comparison
                gt_binary = (gt_mask > 0).astype(float)  # Convert to binary: 0=background, 1=cells
                device_binary = (device_mask > 0).astype(float)  # Convert to binary: 0=background, 1=cells
                
                # Create binary difference mask: 0 where pixels match, 1 where they differ
                diff_binary = (device_binary != gt_binary).astype(float)
                
                # Create RGB image: white background (1,1,1), red differences (1,0,0)
                diff_rgb = np.ones((diff_binary.shape[0], diff_binary.shape[1], 3))
                diff_rgb[:, :, 1] = 1 - diff_binary  # Green channel: 1 for matches, 0 for differences
                diff_rgb[:, :, 2] = 1 - diff_binary  # Blue channel: 1 for matches, 0 for differences
                # Red channel stays 1 everywhere, so differences are red (1,0,0), matches are white (1,1,1)
                
                axes[1, col_idx].imshow(diff_rgb)
                
                # Calculate percentage of differing pixels and cell count comparison
                diff_pixels = int(diff_binary.sum())
                total_pixels = int(diff_binary.size)
                diff_percent = (diff_pixels / total_pixels) * 100
                
                # Format percentage with scientific notation if very small
                if diff_percent < 0.01 and diff_percent > 0:
                    diff_percent_str = f"{diff_percent:.2e}%"
                else:
                    diff_percent_str = f"{diff_percent:.2f}%"
                
                gt_cells = self._count_non_overlapped_cells(gt_mask)  # Count only non-overlapped cells
                device_cells = len(np.unique(device_mask)) - 1  # Subtract 1 for background
                
                axes[1, col_idx].set_title(f"Binary Mask Differences\n({diff_percent_str}, {diff_pixels:,} pixels)\nGT:{gt_cells} vs Dev:{device_cells} cells", 
                                     fontsize=9, fontweight='bold')
                
                # Add black border around bottom row images
                for spine in axes[1, col_idx].spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(3)
                axes[1, col_idx].tick_params(which='both', length=0)  # Hide tick marks
                
                # Keep axis off but borders visible
                axes[1, col_idx].set_xticks([])
                axes[1, col_idx].set_yticks([])
            
            # Extract precision from config_key for title
            precision = config_key.split('_')[-1]  # Get "float32" part from "512x512_float32"
            
            plt.suptitle(f'Cellpose Artificial Inference Ground Truth Comparison - {precision}\nTop: Instance Masks | Bottom: Binary Mask Differences (White=Match, Red=Differ)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            self._save_evaluation_plot(fig, f'ground_truth_comparison_{config_key}')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create ground truth comparison for {config_key}: {e}")

    def _create_ground_truth_comparison_reduced_cpu(self, config_key: str, mask_data: List[Dict[str, Any]]):
        """
        Create ground truth comparison visualization with reduced CPU representation.
        
        If multiple CPU devices are present, only keeps one CPU device to avoid redundancy.
        Shows ground truth and difference plots for each device.
        """
        if not mask_data:
            return
        
        try:
            # Filter mask data to reduce CPU devices to one representative
            filtered_mask_data = self._reduce_cpu_devices(mask_data)
            
            # Check if we actually need the reduced CPU version
            # If the filtering didn't change anything, skip creating duplicate
            if len(filtered_mask_data) == len(mask_data):
                logger.debug(f"No CPU reduction needed for {config_key} - skipping reduced CPU ground truth comparison")
                return
            
            if not filtered_mask_data:
                logger.debug(f"No devices after CPU reduction for ground truth comparison in {config_key}")
                return
            
            # Load the actual ground truth from data directory
            # Parse resolution from config_key (e.g., "512x512_float32" -> (512, 512))
            resolution_str = config_key.split('_')[0]  # Get "512x512" part
            width, height = map(int, resolution_str.split('x'))
            
            # Load ground truth mask from data directory
            dataset = self.config.get('dataset', 'default_dataset')
            data_dir = Path.cwd() / dataset / self.name / "data"
            gt_file = data_dir / f"ground_truth_{width}x{height}.npy"
            
            if not gt_file.exists():
                logger.warning(f"Ground truth file not found: {gt_file}")
                return
            
            gt_data = np.load(gt_file, allow_pickle=True).item()
            if not isinstance(gt_data, dict) or 'ground_truth_mask' not in gt_data:
                logger.warning(f"Invalid ground truth data format in: {gt_file}")
                return
            
            gt_mask = gt_data['ground_truth_mask']
            logger.info(f"  Using actual ground truth from: {gt_file.name} for {config_key} (reduced CPU)")
            
            # Load the original artificial image for the top-left position
            data_dir = Path.cwd() / dataset / self.name / "data"
            img_file = data_dir / f"artificial_image_{width}x{height}.npy"
            
            if not img_file.exists():
                logger.warning(f"Original image file not found: {img_file}")
                return
            
            original_image = np.load(img_file)
            logger.info(f"  Using original image from: {img_file.name} for {config_key} (reduced CPU)")
            
            # Normalize ground truth mask for consistent visualization
            gt_mask_normalized = self._normalize_instance_mask(gt_mask)
            gt_max_instances = np.max(gt_mask_normalized) if np.max(gt_mask_normalized) > 0 else 1
            
            # Create comparison plot with ground truth column
            import matplotlib.pyplot as plt
            
            n_devices = len(filtered_mask_data)
            # Add 1 extra column for ground truth
            fig, axes = plt.subplots(2, n_devices + 1, figsize=(4*(n_devices + 1), 8))
            if n_devices == 0:
                return
            
            # First column, top row: Original input image
            axes[0, 0].imshow(original_image, cmap='gray')
            axes[0, 0].set_title("Original Image\nInput", fontsize=10, fontweight='bold')
            axes[0, 0].axis('off')
            
            # First column, bottom row: Ground truth mask
            axes[1, 0].imshow(gt_mask_normalized, cmap='viridis', vmin=0, vmax=gt_max_instances)
            axes[1, 0].set_title("Ground Truth\nReference", fontsize=9, fontweight='bold')
            axes[1, 0].axis('off')
            
            for idx, mask_entry in enumerate(filtered_mask_data):
                device_mask = self._load_mask(mask_entry)
                if device_mask is None:
                    continue
                
                # Adjust column index to account for ground truth column
                col_idx = idx + 1
                
                # Get device flavour information for consistent naming
                device_flavour = mask_entry.get('flavour', 'unknown')
                device_type = mask_entry['device_type']
                device_model = mask_entry.get('device_model', mask_entry['device_name'])
                
                # Create title: remove flavour only for CPU devices in reduced CPU version
                if device_type.lower() == 'cpu':
                    device_title = device_model
                else:
                    flavour_name = self._create_flavour_name(device_flavour, device_type)
                    device_title = f"{flavour_name}:\n{device_model}"
                
                # Normalize device mask for consistent visualization
                device_mask_normalized = self._normalize_instance_mask(device_mask)
                device_max_instances = np.max(device_mask_normalized) if np.max(device_mask_normalized) > 0 else 1
                
                # Top row: Original masks with improved titles and dynamic colormap range
                axes[0, col_idx].imshow(device_mask_normalized, cmap='viridis', vmin=0, vmax=device_max_instances)
                axes[0, col_idx].set_title(device_title, fontsize=10, fontweight='bold')
                axes[0, col_idx].axis('off')
                
                # Bottom row: Binary difference visualization with black borders
                # Convert instance masks to binary masks for meaningful comparison
                gt_binary = (gt_mask > 0).astype(float)  # Convert to binary: 0=background, 1=cells
                device_binary = (device_mask > 0).astype(float)  # Convert to binary: 0=background, 1=cells
                
                # Create binary difference mask: 0 where pixels match, 1 where they differ
                diff_binary = (device_binary != gt_binary).astype(float)
                
                # Create RGB image: white background (1,1,1), red differences (1,0,0)
                diff_rgb = np.ones((diff_binary.shape[0], diff_binary.shape[1], 3))
                diff_rgb[:, :, 1] = 1 - diff_binary  # Green channel: 1 for matches, 0 for differences
                diff_rgb[:, :, 2] = 1 - diff_binary  # Blue channel: 1 for matches, 0 for differences
                # Red channel stays 1 everywhere, so differences are red (1,0,0), matches are white (1,1,1)
                
                axes[1, col_idx].imshow(diff_rgb)
                
                # Calculate percentage of differing pixels and cell count comparison
                diff_pixels = int(diff_binary.sum())
                total_pixels = int(diff_binary.size)
                diff_percent = (diff_pixels / total_pixels) * 100
                
                # Format percentage with scientific notation if very small
                if diff_percent < 0.01 and diff_percent > 0:
                    diff_percent_str = f"{diff_percent:.2e}%"
                else:
                    diff_percent_str = f"{diff_percent:.2f}%"
                
                gt_cells = self._count_non_overlapped_cells(gt_mask)  # Count only non-overlapped cells
                device_cells = len(np.unique(device_mask)) - 1  # Subtract 1 for background
                
                axes[1, col_idx].set_title(f"Binary Mask Differences\n({diff_percent_str}, {diff_pixels:,} pixels)\nGT:{gt_cells} vs Dev:{device_cells} cells", 
                                     fontsize=9, fontweight='bold')
                
                # Add black border around bottom row images
                for spine in axes[1, col_idx].spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(3)
                axes[1, col_idx].tick_params(which='both', length=0)  # Hide tick marks
                
                # Keep axis off but borders visible
                axes[1, col_idx].set_xticks([])
                axes[1, col_idx].set_yticks([])
            
            # Extract precision from config_key for title
            precision = config_key.split('_')[-1]  # Get "float32" part from "512x512_float32"
            
            plt.suptitle(f'Cellpose Artificial Inference Ground Truth Comparison (Reduced CPU) - {precision}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            self._save_evaluation_plot(fig, f'ground_truth_comparison_reduced_cpu_{config_key}')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create reduced CPU ground truth comparison for {config_key}: {e}")
    
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
                    # Normalize mask for consistent visualization
                    normalized_mask = self._normalize_instance_mask(mask)
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
                    if i == j or i > j:
                        pass  # Keep these visible            # Extract precision from config_key for title
            precision = config_key.split('_')[-1]
            plt.suptitle(f'Cellpose Device-to-Device Comparison Matrix - {precision}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            self._save_evaluation_plot(fig, f'device_comparison_matrix_{config_key}')
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
                    if i == j or i > j:
                        pass  # Keep these visible
            
            # Extract precision from config_key for title
            precision = config_key.split('_')[-1]
            plt.suptitle(f'Cellpose Device-to-Device Comparison Matrix (Reduced CPU) - {precision}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            self._save_evaluation_plot(fig, f'device_comparison_matrix_reduced_cpu_{config_key}')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create reduced CPU device comparison matrix for {config_key}: {e}")

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
                return np.load(mask_path)
            else:
                logger.warning(f"Mask file not found: {mask_path}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load mask: {e}")
            return None
    
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
