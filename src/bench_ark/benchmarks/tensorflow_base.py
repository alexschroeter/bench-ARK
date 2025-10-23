"""Tensorflow-specific benchmark base class."""

import logging
from typing import List, Dict, Any, Optional, Union
from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class TensorFlowBenchmark(BaseBenchmark):
    """Base class for TensorFlow-based benchmarks."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.tensorflow = None
        self.tensorflow_available = False
        self._initialize_tensorflow()
        super().__init__(name, config)

    def _initialize_tensorflow(self) -> None:
        """Initialize TensorFlow and extensions."""
        try:
            import tensorflow
            self.tensorflow = tensorflow
            self.tensorflow_available = True
            logger.debug("TensorFlow available for device detection")
            
            # Try to initialize Intel extension
            try:
                import intel_extension_for_tensorflow as ipex
                self.ipex = ipex
                logger.debug("Intel Extension for TensorFlow loaded")
            except ImportError:
                self.ipex = None
                logger.debug("Intel Extension for TensorFlow not available")
            
        except ImportError as e:
            logger.warning(f"TensorFlow not available: {e}")

    def _get_available_devices(self) -> List[Dict[str, Any]]:
        """Get a list of available devices for TensorFlow benchmarking."""
        devices = []
        
        # CPU device
        devices.append({
            'name': 'CPU',
            'type': 'cpu',
            'id': 'cpu',
            'memory': None,
            'framework': 'tensorflow',
            'properties': {}
        })
        if not self.tensorflow_available:
            logger.debug("TensorFlow not available, returning CPU only")
            return devices
        
        logger.debug(f"arkitekt_flavour: {self.config.get('arkitekt_flavour')}")
        
        if self.config.get("arkitekt_flavour") == "nvidia_gpu" or self.config.get("arkitekt_flavour") == "amd_gpu":
            if self.tensorflow.test.is_gpu_available():
                physical_devices = self.tensorflow.config.list_physical_devices('GPU')
                for i in range(len(physical_devices)):
                    props = self.tensorflow.config.experimental.get_device_details(physical_devices[i])
                    devices.append({
                        'name': f'{props["device_name"]}',
                        'type': 'cuda' if self.config.get("arkitekt_flavour") == 'nvidia_gpu' else 'rocm',
                        'id': i,
                        # 'memory': props.total_memory / 1024**3,  # GB
                        'framework': 'tensorflow',
                        # 'properties': {
                        #     'name': props.name,
                        #     'major': props.major,
                        #     'minor': props.minor,
                        #     'multi_processor_count': props.multi_processor_count,
                        #     'total_memory': props.total_memory
                        # }
                    })
                    
        if self.config.get("arkitekt_flavour") == "intel_gpu":
            logger.debug("Intel GPU flavour detected, checking for XPU devices")
            # Intel XPU devices
            if self.ipex and hasattr(self.tensorflow, 'xpu'):
                logger.debug("Intel extension and XPU available")
                try:
                    if self.tensorflow.test.is_gpu_available():
                        physical_devices = self.tensorflow.config.list_physical_devices('GPU')
                        logger.debug(f"XPU is available, device count: {len(physical_devices)}")
                        for i in range(len(physical_devices)):
                            props = self.tensorflow.config.experimental.get_device_details(physical_devices[i])
                            # logger.debug(f"Adding XPU device {i}: {props["device_name"]}")
                            devices.append({
                                'name': f'{props["device_name"]}',
                                'type': 'xpu',
                                'id': i,
                                # 'memory': props.total_memory / 1024**3,  # GB
                                'framework': 'tensorflow',
                                'properties': {
                                    # 'name': props.name,
                                    # 'platform_name': getattr(props, 'platform_name', 'Unknown'),
                                    # 'type': getattr(props, 'type', 'gpu'),
                                    # 'driver_version': getattr(props, 'driver_version', 'Unknown'),
                                    # 'max_compute_units': getattr(props, 'max_compute_units', 0),
                                    # 'gpu_eu_count': getattr(props, 'gpu_eu_count', 0),
                                    # 'gpu_subslice_count': getattr(props, 'gpu_subslice_count', 0),
                                    # 'max_work_group_size': getattr(props, 'max_work_group_size', 0),
                                    # 'total_memory': props.total_memory,
                                    # 'has_fp16': getattr(props, 'has_fp16', False),
                                    # 'has_fp64': getattr(props, 'has_fp64', False),
                                    # 'has_atomic64': getattr(props, 'has_atomic64', False)
                                }
                            })
                    else:
                        logger.debug("XPU is not available")
                except Exception as e:
                    logger.debug(f"Intel XPU detection failed: {e}")
            else:
                logger.debug(f"Intel extension available: {self.ipex is not None}, XPU hasattr: {hasattr(self.tensorflow, 'xpu') if self.tensorflow else False}")
        
        logger.debug(f"Total devices detected: {len(devices)}")
        return devices
        
    def _run_benchmark(self) -> Dict[str, Any]:
        """
        Abstract method for running the benchmark.
        Subclasses must override this method to implement specific benchmark logic.
        """
        raise NotImplementedError("Subclasses must implement _run_benchmark method")
