"""PyTorch-specific benchmark base class."""

import logging
from typing import List, Dict, Any, Optional, Union
from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class PyClesperantoBenchmark(BaseBenchmark):
    """Base class for OpenCL-based benchmarks."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.opencl = None
        self.opencl_available = False
        self._initialize_opencl()
        super().__init__(name, config)

    def _initialize_opencl(self) -> None:
        """Initialize OpenCL and extensions."""
        try:
            import pyclesperanto_prototype
            self.opencl = pyclesperanto_prototype
            self.opencl_available = True
            logger.debug("OpenCL available for device detection")

        except ImportError as e:
            logger.warning(f"OpenCL not available: {e}")

    def _get_available_devices(self) -> List[Dict[str, Any]]:
        """Get a list of available devices for OpenCL benchmarking."""
        devices = []
        
        # CPU device OpenCL has no CPU fallback like other Frameworks so the 
        # default CPU doesnt need to be added.
        #
        # devices.append({
        #     'name': 'CPU',
        #     'type': 'cpu',
        #     'id': 'cpu',
        #     'memory': None,
        #     'framework': 'pyclesperanto',
        #     'properties': {}
        # })
        if not self.opencl_available:
            logger.debug("OpenCL not available, returning CPU only")
            return devices
        
        logger.debug(f"arkitekt_flavour: {self.config.get('arkitekt_flavour')}")
        
        if self.config.get("arkitekt_flavour") == "nvidia_gpu" or self.config.get("arkitekt_flavour") == "amd_gpu" or self.config.get("arkitekt_flavour") == "intel_gpu":
            if self.opencl_available:
                # Get the list of device names and iterate over them with indices
                device_names = self.opencl.available_device_names()
                for i in range(len(device_names)):
                    props = self.opencl.select_device(device_index=i)
                    # Determine device type based on arkitekt_flavour
                    flavour = self.config.get("arkitekt_flavour")
                    if flavour == 'nvidia_gpu':
                        device_type = 'cuda'
                    elif flavour == 'amd_gpu':
                        device_type = 'rocm'
                    elif flavour == 'intel_gpu':
                        device_type = 'xpu'
                    else:
                        device_type = 'opencl'  # fallback
                    
                    devices.append({
                        'name': f'{props.name}',
                        'type': device_type,
                        'framework': 'pytorch',
                        'id': i
                    })
        
        logger.debug(f"Total devices detected: {len(devices)}")
        return devices
        
    def _run_benchmark(self) -> Dict[str, Any]:
        """
        Abstract method for running the benchmark.
        Subclasses must override this method to implement specific benchmark logic.
        """
        raise NotImplementedError("Subclasses must implement _run_benchmark method")
