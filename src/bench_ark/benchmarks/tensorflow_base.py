"""TensorFlow-specific benchmark base class."""

import logging
from typing import List, Dict, Any, Optional, Union
from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class TensorFlowBenchmark(BaseBenchmark):
    """Base class for TensorFlow-based benchmarks."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.tf = None
        self.tf_available = False
        self._initialize_tensorflow()
    
    def _initialize_tensorflow(self) -> None:
        """Initialize TensorFlow."""
        try:
            import tensorflow as tf
            self.tf = tf
            self.tf_available = True
            self.logger.info("TensorFlow available for device detection")
            
            # Suppress TensorFlow warnings by default
            tf.get_logger().setLevel('ERROR')
            
        except ImportError as e:
            self.logger.warning(f"TensorFlow not available: {e}")
    
    def get_framework_name(self) -> str:
        """Get the framework name."""
        return "tensorflow"
    
    def get_supported_devices(self) -> List[str]:
        """Get TensorFlow-supported devices."""
        devices = []
        
        if not self.tf_available:
            devices.append("CPU")
            return devices
        
        # Get all available devices
        physical_devices = self.tf.config.list_physical_devices()
        
        for device in physical_devices:
            if device.device_type == 'CPU':
                devices.append("CPU")
            elif device.device_type == 'GPU':
                # Extract GPU information
                device_name = device.name.split('/')[-1]  # e.g., 'GPU:0'
                try:
                    # Try to get GPU details
                    gpu_details = self.tf.config.experimental.get_device_details(device)
                    if gpu_details and 'device_name' in gpu_details:
                        gpu_name = gpu_details['device_name']
                        devices.append(f"TF {device_name} ({gpu_name})")
                    else:
                        devices.append(f"TF {device_name}")
                except Exception:
                    devices.append(f"TF {device_name}")
        
        return devices
    
    def get_device_details(self) -> List[Dict[str, Any]]:
        """Get detailed TensorFlow device information."""
        devices = []
        
        if not self.tf_available:
            devices.append({
                'name': 'CPU',
                'type': 'cpu',
                'id': 'cpu',
                'memory': None,
                'framework': 'tensorflow',
                'properties': {}
            })
            return devices
        
        physical_devices = self.tf.config.list_physical_devices()
        
        for device in physical_devices:
            device_info = {
                'name': device.name,
                'type': device.device_type.lower(),
                'id': device.name.split('/')[-1],
                'memory': None,
                'framework': 'tensorflow',
                'properties': {}
            }
            
            if device.device_type == 'CPU':
                device_info['name'] = 'CPU'
                device_info['id'] = 'cpu'
            elif device.device_type == 'GPU':
                try:
                    # Get GPU details
                    gpu_details = self.tf.config.experimental.get_device_details(device)
                    if gpu_details:
                        device_info['properties'] = gpu_details
                        if 'device_name' in gpu_details:
                            device_name = gpu_details['device_name']
                            device_info['name'] = f"TF {device.name.split('/')[-1]} ({device_name})"
                        
                        # Try to get memory info
                        if 'memory_limit' in gpu_details:
                            device_info['memory'] = gpu_details['memory_limit'] / 1024**3  # GB
                except Exception as e:
                    self.logger.debug(f"Could not get GPU details: {e}")
                    device_info['name'] = f"TF {device.name.split('/')[-1]}"
            
            devices.append(device_info)
        
        return devices
    
    def is_device_available(self, device_identifier: str) -> bool:
        """Check if a TensorFlow device is available."""
        if not self.tf_available:
            return device_identifier.upper() == "CPU"
        
        available_devices = self.get_supported_devices()
        return device_identifier in available_devices
    
    def prepare_device(self, device_identifier: str) -> Union[str, Any]:
        """Prepare a TensorFlow device context."""
        if not self.tf_available:
            return '/CPU:0'
        
        if device_identifier.upper() == "CPU":
            return '/CPU:0'
        
        # For GPU devices, extract the device path
        if "TF GPU:" in device_identifier:
            try:
                import re
                match = re.search(r'TF (GPU:\d+)', device_identifier)
                if match:
                    return f'/{match.group(1)}'
            except Exception:
                pass
        
        # Fallback to CPU
        self.logger.warning(f"Could not prepare device {device_identifier}, falling back to CPU")
        return '/CPU:0'
    
    def get_device_memory_info(self, device_identifier: str) -> Dict[str, Any]:
        """Get memory information for a TensorFlow device."""
        memory_info = {'total': None, 'allocated': None, 'free': None}
        
        if not self.tf_available:
            return memory_info
        
        if "GPU" in device_identifier.upper():
            try:
                # TensorFlow memory info is more complex to get
                # This is a simplified version
                physical_devices = self.tf.config.list_physical_devices('GPU')
                if physical_devices:
                    # Get the first GPU for simplicity
                    gpu_details = self.tf.config.experimental.get_device_details(physical_devices[0])
                    if gpu_details and 'memory_limit' in gpu_details:
                        memory_info['total'] = gpu_details['memory_limit']
            except Exception as e:
                self.logger.debug(f"Could not get TensorFlow memory info: {e}")
        
        return memory_info
