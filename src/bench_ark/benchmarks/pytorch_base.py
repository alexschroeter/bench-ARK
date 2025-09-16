"""PyTorch-specific benchmark base class."""

import logging
from typing import List, Dict, Any, Optional, Union
from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class PyTorchBenchmark(BaseBenchmark):
    """Base class for PyTorch-based benchmarks."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.torch = None
        self.torch_available = False
        self._initialize_pytorch()
        super().__init__(name, config)

    def _initialize_pytorch(self) -> None:
        """Initialize PyTorch and extensions."""
        try:
            import torch
            self.torch = torch
            self.torch_available = True
            logger.debug("PyTorch available for device detection")
            
            # Try to initialize Intel extension
            try:
                import intel_extension_for_pytorch as ipex
                self.ipex = ipex
                logger.debug("Intel Extension for PyTorch loaded")
            except ImportError:
                self.ipex = None
                logger.debug("Intel Extension for PyTorch not available")
            
        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}")
    
    def _get_available_devices(self) -> List[Dict[str, Any]]:
        """Get a list of available devices for PyTorch benchmarking."""
        devices = []
        
        # CPU device
        devices.append({
            'name': 'CPU',
            'type': 'cpu',
            'id': 'cpu',
            'memory': None,
            'framework': 'pytorch',
            'properties': {}
        })
        if not self.torch_available:
            logger.debug("PyTorch not available, returning CPU only")
            return devices
        
        logger.debug(f"arkitekt_flavour: {self.config.get('arkitekt_flavour')}")
        
        if self.config.get("arkitekt_flavour") == "nvidia_gpu" or self.config.get("arkitekt_flavour") == "amd_gpu":
            if self.torch.cuda.is_available():
                for i in range(self.torch.cuda.device_count()):
                    props = self.torch.cuda.get_device_properties(i)
                    devices.append({
                        'name': f'{props.name}',
                        'type': 'cuda' if self.config.get("arkitekt_flavour") == 'nvidia_gpu' else 'rocm',
                        'id': i,
                        'memory': props.total_memory / 1024**3,  # GB
                        'framework': 'pytorch',
                        'properties': {
                            'name': props.name,
                            'major': props.major,
                            'minor': props.minor,
                            'multi_processor_count': props.multi_processor_count,
                            'total_memory': props.total_memory
                        }
                    })
                    
        if self.config.get("arkitekt_flavour") == "intel_gpu":
            logger.debug("Intel GPU flavour detected, checking for XPU devices")
            # Intel XPU devices
            if self.ipex and hasattr(self.torch, 'xpu'):
                logger.debug("Intel extension and XPU available")
                try:
                    if self.torch.xpu.is_available():
                        logger.debug(f"XPU is available, device count: {self.torch.xpu.device_count()}")
                        for i in range(self.torch.xpu.device_count()):
                            props = self.torch.xpu.get_device_properties(i)
                            logger.debug(f"Adding XPU device {i}: {props.name}")
                            devices.append({
                                'name': f'{props.name}',
                                'type': 'xpu',
                                'id': i,
                                'memory': props.total_memory / 1024**3,  # GB
                                'framework': 'pytorch',
                                'properties': {
                                    'name': props.name,
                                    'platform_name': getattr(props, 'platform_name', 'Unknown'),
                                    'type': getattr(props, 'type', 'gpu'),
                                    'driver_version': getattr(props, 'driver_version', 'Unknown'),
                                    'max_compute_units': getattr(props, 'max_compute_units', 0),
                                    'gpu_eu_count': getattr(props, 'gpu_eu_count', 0),
                                    'gpu_subslice_count': getattr(props, 'gpu_subslice_count', 0),
                                    'max_work_group_size': getattr(props, 'max_work_group_size', 0),
                                    'total_memory': props.total_memory,
                                    'has_fp16': getattr(props, 'has_fp16', False),
                                    'has_fp64': getattr(props, 'has_fp64', False),
                                    'has_atomic64': getattr(props, 'has_atomic64', False)
                                }
                            })
                    else:
                        logger.debug("XPU is not available")
                except Exception as e:
                    logger.debug(f"Intel XPU detection failed: {e}")
            else:
                logger.debug(f"Intel extension available: {self.ipex is not None}, XPU hasattr: {hasattr(self.torch, 'xpu') if self.torch else False}")
        
        logger.debug(f"Total devices detected: {len(devices)}")
        return devices
        
    def _run_benchmark(self) -> Dict[str, Any]:
        """
        Abstract method for running the benchmark.
        Subclasses must override this method to implement specific benchmark logic.
        """
        raise NotImplementedError("Subclasses must implement _run_benchmark method")
