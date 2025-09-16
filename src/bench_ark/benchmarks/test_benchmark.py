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


class TestBenchmark(PyTorchBenchmark):
    """
    A simple artificial benchmark for testing the bench-ARK framework.
    
    This benchmark performs basic mathematical operations to simulate
    workload and test device detection, benchmark execution, and result handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        logger.debug("TestBenchmark initialization starting...")
        super().__init__("test_benchmark", config)
        logger.debug("TestBenchmark initialization completed.")
        
    def _run_benchmark(self) -> Dict[str, Any]:
        logger.info(f"Starting Test Benchmark on {len(self.device_manager._list_available_devices())} devices")
        logger.debug("This is a DEBUG message from test_benchmark")
        logger.debug("Performing artificial calculations...")
        result = {"this is a result": "really", "calculation": 42 * 1337}
        logger.debug(f"Benchmark completed with result: {result}")
        return result
    
    def _evaluate_benchmark(self) -> None:
        """
        Evaluate the benchmark results and create performance plots.
        """
        from ..core.visualizations import PerformancePlotter
        
        # Log basic results
        for result in self.all_results:
            device_name = result["devices"][0]["name"] if result.get("devices") and len(result["devices"]) > 0 else "Unknown"
            flavour = result.get("flavour", "Unknown")
            calculation = result.get("results", {}).get("calculation", "Unknown")
            logger.info(f"on device {device_name} with flavour {flavour} the result was {calculation}.")
        
        # Generate fake data for plotting demonstration
        fake_plotting_data = self._generate_fake_plotting_data()
        
        try:
            # Create performance plotter with fake data
            plotter = PerformancePlotter(fake_plotting_data)
            
            # Create plot path following pattern: <dataset>/<benchmark_name>/<timestamp>_plots/performance_benchmark
            from datetime import datetime
            dataset = self.config.get('dataset', 'default_dataset')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"{dataset}/{self.name}/{timestamp}_plots/performance_benchmark"
            
            plotter.store_plot(plot_path)
            logger.info(f"Performance plot saved to: {plot_path}")
            
        except ImportError as e:
            logger.warning(f"Could not create performance plot: {e}")
        except Exception as e:
            logger.error(f"Error creating performance plot: {e}")

    def _convert_results_for_plotting(self) -> list:
        """
        Convert self.all_results into a format suitable for PerformancePlotter.
        This method should extract relevant metrics from the benchmark results
        and structure them in a way that PerformancePlotter can use.
        """
        pass
    
    def _generate_fake_plotting_data(self) -> list:
        """
        Generate fake benchmark data for testing plotting functionality.
        
        Returns:
            List of fake benchmark results that can be used by PerformancePlotter
        """
        import random
        import numpy as np
        
        fake_data = []
        
        # Simulate different devices and vendors
        devices = [
            {'name': 'NVIDIA GPU 0', 'type': 'cuda', 'vendor': 'nvidia', 'flavour': 'nvidia_gpu'},
            {'name': 'Intel XPU 0', 'type': 'xpu', 'vendor': 'intel', 'flavour': 'intel_gpu'},
            {'name': 'AMD GPU 0', 'type': 'hip', 'vendor': 'amd', 'flavour': 'amd_gpu'},
            {'name': 'CPU', 'type': 'cpu', 'vendor': 'intel', 'flavour': 'vanilla'},
        ]
        
        # Generate fake performance data for different image sizes
        image_sizes = [256*256, 512*512, 1024*1024, 2048*2048, 4096*4096]
        
        for device in devices:
            # Simulate base execution time for this device
            if device['type'] == 'cpu':
                base_time = random.uniform(2.0, 5.0)  # CPU is slower
            else:
                base_time = random.uniform(0.1, 1.0)  # GPU is faster
            
            # Create result for simple benchmark (no complex metrics)
            simple_result = {
                'device_name': device['name'],
                'device_type': device['type'],
                'flavour': device['flavour'],
                'execution_time': base_time + random.uniform(-0.1, 0.1),
                'results': {
                    'calculation': 42 * 1337 + random.randint(-100, 100)
                }
            }
            fake_data.append(simple_result)
            
            # Create complex benchmark results with performance metrics
            performance_metrics = {}
            
            for img_size in image_sizes:
                # Simulate inference time scaling with image size
                size_factor = img_size / (256*256)  # Normalize to smallest size
                inference_time = base_time * (size_factor ** 0.7)  # Sub-linear scaling
                inference_time += random.uniform(-inference_time*0.1, inference_time*0.1)  # Add noise
                
                # Ensure positive time
                inference_time = max(0.001, inference_time)
                
                width = int(np.sqrt(img_size))
                height = width
                
                metric_key = f"{width}x{height}_float32"
                performance_metrics[metric_key] = {
                    'success': True,
                    'resolution': f"{width}x{height}",
                    'image_size_pixels': img_size,
                    'inference_time': inference_time,
                    'throughput_px_per_sec': img_size / inference_time if inference_time > 0 else 0,
                    'precision_used': 'float32',
                    'num_cells': random.randint(50, 500)
                }
            
            # Create complex benchmark result
            complex_result = {
                'device_name': device['name'],
                'device_type': device['type'],
                'flavour': device['flavour'],
                'execution_time': sum(m['inference_time'] for m in performance_metrics.values()),
                'performance_metrics': performance_metrics,
                'results': {
                    'tests_completed': len(performance_metrics),
                    'tests_failed': 0
                }
            }
            fake_data.append(complex_result)
        
        logger.debug(f"Generated {len(fake_data)} fake data points for plotting")
        return fake_data