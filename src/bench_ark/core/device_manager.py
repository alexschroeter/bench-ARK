"""
Each benchmark has its own way of getting the device information.
Because a benchmark is bound to the framework it is using to run.
This means that the device might be identified differently across different frameworks.
The Device manager adds to the information passed by the framework.

ToDo: 
- Implement fetching of manual information for the device.
- Implement fetching of external information for the device.
- If bench-ARK becomes it's own service as part of the Arkitekt framework,
  it might be nice to have a unified way of accessing device information across different benchmarks
  and a way to identify the same device across different benchmarks. This would
  allow for easier comparison and analysis of benchmark results across versions and flavours.
"""

import logging
import os
import subprocess
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class Device:
    def __init__(self, **properties):
        for key, value in properties.items():
            setattr(self, key, value)

    def __repr__(self):
        # Show name and any additional properties
        return f"Device({self.name}"

    def _get_configuration_information(self):
        """
        ToDo: Implement fetching of manual information for the device.
              Most likely some form of device_information.yaml
        """
        pass
    
    def _get_external_information(self):
        """
        Implement fetching of external information for the device.
        Most likely reading some docker/pci/smi information
        """
        # Read ARKITEKT_FLAVOUR environment variable and add it to device properties
        flavour = os.getenv('ARKITEKT_FLAVOUR', None)
        self.arkitekt_flavour = flavour
        logger.debug(f"Added flavour '{flavour}' to device {getattr(self, 'name', 'unknown')}")
        
        # Get CPU information using lscpu
        self._get_cpu_information()
    
    def _get_cpu_information(self):
        """
        Fetch CPU information using lscpu command and add relevant details to device properties.
        """
        try:
            # Run lscpu command to get CPU information
            result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                cpu_info = {}
                lines = result.stdout.strip().split('\n')
                
                # Parse lscpu output for key information
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Store relevant CPU information
                        if key in ['Architecture', 'Model name', 'CPU(s)', 'Thread(s) per core', 
                                  'Core(s) per socket', 'Socket(s)', 'CPU MHz', 'CPU max MHz', 
                                  'CPU min MHz', 'L1d cache', 'L1i cache', 'L2 cache', 'L3 cache',
                                  'Vendor ID', 'CPU family', 'Model', 'Stepping', 'Flags']:
                            # Clean up key names for attribute storage
                            attr_key = key.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                            cpu_info[attr_key] = value
                
                # Add CPU information to device properties
                self.cpu_info = cpu_info
                
                # Add some key CPU properties directly as device attributes for easy access
                if 'model_name' in cpu_info:
                    self.cpu_model = cpu_info['model_name']
                if 'cpus' in cpu_info:
                    self.cpu_count = cpu_info['cpus']
                if 'architecture' in cpu_info:
                    self.cpu_architecture = cpu_info['architecture']
                
                logger.debug(f"Added CPU information to device {getattr(self, 'name', 'unknown')}: "
                           f"Model: {cpu_info.get('model_name', 'Unknown')}, "
                           f"Cores: {cpu_info.get('cpus', 'Unknown')}, "
                           f"Architecture: {cpu_info.get('architecture', 'Unknown')}")
            else:
                logger.warning(f"lscpu command failed with return code {result.returncode}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning("lscpu command timed out")
        except FileNotFoundError:
            logger.warning("lscpu command not found - CPU information will not be available")
        except Exception as e:
            logger.warning(f"Failed to get CPU information: {str(e)}")

class DeviceManager:
    """Manages device detection and coordination across different benchmark frameworks."""

    def __init__(self):
        """Initialize DeviceManager without any specific context."""
        self.available_devices = []

    def _set_available_devices(self, device_dict: List[Dict[str, Any]]) -> None:
        self.available_devices = self._update_device_properties([Device(**props) for props in device_dict])
        self._update_device_properties(self.available_devices)

    def _list_available_devices(self) -> List[Device]:
        """Returns a list of all available devices."""
        return self.available_devices
    
    def _update_device_properties(self, devices: List[Device]) -> List[Device]:
        """
        Update device properties based on predefined rules or external data.
        This is a placeholder for actual implementation.
        """
        # Example: Add a dummy property to each device
        for device in devices:
            device._get_configuration_information()
            device._get_external_information()
        return devices
