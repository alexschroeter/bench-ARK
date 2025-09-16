"""
This module contains classes for visualizations.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# For plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    import warnings
    warnings.warn("Matplotlib and/or NumPy not available - plots cannot be generated")

logger = logging.getLogger(__name__)


class PerformancePlotter:
    """
    A class for plotting performance metrics of algorithms.
    """
    
    def __init__(self, data: List[Dict[str, Any]]):
        if not PLOTTING_AVAILABLE:
            raise ImportError("Matplotlib and NumPy are required for plotting")
        
        self.data = data
        
        # Vendor colors and styles (now flavour-based)
        self.flavour_colors = {
            'nvidia': '#66AA66',   # Muted green for NVIDIA
            'intel': '#6699CC',    # Muted blue for Intel
            'amd': '#CC6666',      # Muted red for AMD
            'unknown': '#666666'   # Gray for unknown flavours
        }
        
        # Device type styles within flavours
        self.device_styles = {
            'cpu': {'linestyle': ':', 'marker': '^', 'markersize': 8},   # Dotted line, triangle for CPU
            'gpu': {'linestyle': '-', 'marker': 'o', 'markersize': 6},   # Solid line, circle for GPU
            'cuda': {'linestyle': '-', 'marker': 'o', 'markersize': 6},  # CUDA is GPU - solid line
            'hip': {'linestyle': '-', 'marker': 'o', 'markersize': 6},   # HIP is GPU - solid line
            'rocm': {'linestyle': '-', 'marker': 'o', 'markersize': 6},  # ROCm is GPU - solid line
            'xpu': {'linestyle': '-', 'marker': 'o', 'markersize': 6},   # XPU is GPU - solid line
            'unknown': {'linestyle': '--', 'marker': 's', 'markersize': 6}, # Dashed line, square for unknown
        }
        
        self.plot = self._create_plot()

    def _create_plot(self):
        """
        Create performance comparison plot based on benchmark results data.
        
        Expected data format:
        [
            {
                'device_name': 'NVIDIA GPU 0', 
                'device_type': 'cuda',
                'device_model': 'NVIDIA GeForce RTX 3080',
                'flavour': 'nvidia_gpu',
                'flavour_name': 'NVIDIA CUDA',
                'vendor': 'nvidia',
                'execution_time': 1.23,
                'performance_metrics': {...},
                'batch_sizes': [1, 2, 4, 8],  # Optional for simple plots
                'times': [0.1, 0.15, 0.25, 0.4],  # Optional for simple plots
                'benchmark_title': 'Custom Benchmark Title',  # Optional
                'precision': 'float32',  # Required for precision-based plots
                ...
            },
            ...
        ]
        
        Returns:
            matplotlib Figure object or dict of Figure objects (one per precision)
        """
        if not self.data:
            logger.warning("No data provided for plotting")
            return None
        
        logger.info(f"Creating performance plot with {len(self.data)} data points")
        
        # Extract and process plot data
        plot_data = self._extract_plot_data()
        
        if not plot_data:
            logger.warning("No valid performance data found for plotting")
            return None
        
        # Group data by precision first
        precision_groups = self._group_data_by_precision(plot_data)
        
        # Always return dict of figures to ensure precision-based filenames
        figures = {}
        for precision, precision_data in precision_groups.items():
            figures[precision] = self._create_single_precision_plot(precision_data, precision)
        
        # For backward compatibility, if there's only one precision, 
        # still return dict but this ensures store_plot handles it correctly
        return figures
    
    def _group_data_by_precision(self, plot_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group data points by precision."""
        precision_groups = {}
        
        for point in plot_data:
            precision = point.get('precision', 'unknown')
            if precision not in precision_groups:
                precision_groups[precision] = []
            precision_groups[precision].append(point)
        
        logger.info(f"Found {len(precision_groups)} precision groups: {list(precision_groups.keys())}")
        return precision_groups
    
    def _create_single_precision_plot(self, plot_data: List[Dict[str, Any]], precision: str):
        """Create a single plot for a specific precision."""
        # Create plot configuration from data
        plot_config = self._create_plot_config(plot_data)
        plot_config['precision'] = precision
        
        # Update title to include precision
        base_title = plot_config.get('title', 'Performance Comparison')
        plot_config['title'] = f"{base_title} - {precision.upper()}"
        
        # Create the plot
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=plot_config.get('figsize', (12, 8)))
        
        # Group data by flavour and device for plotting
        flavour_data = self._group_data_by_flavour(plot_data)
        
        # Plot each flavour
        for flavour_key, flavour_devices in flavour_data.items():
            self._plot_flavour_performance(ax, flavour_key, flavour_devices, plot_config)
        
        # Customize the plot
        self._customize_flavour_plot(ax, flavour_data, plot_config)
        
        plt.tight_layout()
        return fig
    
    def _extract_plot_data(self) -> List[Dict[str, Any]]:
        """Extract plottable data from benchmark results."""
        plot_data = []
        
        for result in self.data:
            # Extract basic information
            execution_time = result.get('execution_time', 0.0)
            device_name = result.get('device_name', 'Unknown')
            device_type = result.get('device_type', 'unknown')
            flavour = result.get('flavour', 'unknown')
            
            # Extract vendor from flavour or device name
            vendor = self._extract_vendor(flavour, device_name)
            
            # Try to get additional metrics from performance_metrics if available
            perf_metrics = result.get('performance_metrics', {})
            
            # Create base data point
            data_point = {
                'device_name': device_name,
                'device_type': device_type,
                'vendor': vendor,
                'flavour': flavour,
                'execution_time': execution_time,
                'calculation_result': result.get('results', {}).get('calculation', 0),
                # Add device model if available
                'device_model': result.get('device_model', device_name),
                # Add flavour name if available
                'flavour_name': result.get('flavour_name', f"{vendor.title()} {device_type.upper()}"),
                # Add batch_sizes and times if available for simple plots
                'batch_sizes': result.get('batch_sizes', []),
                'times': result.get('times', []),
                # Add precision for precision-based plotting
                'precision': result.get('precision', 'float32')  # Default to float32 if not specified
            }
            
            # Preserve benchmark_title if it exists
            if 'benchmark_title' in result:
                data_point['benchmark_title'] = result['benchmark_title']
            
            # If we have complex performance metrics (like from real benchmarks)
            if isinstance(perf_metrics, dict) and perf_metrics:
                # Extract image size and inference metrics if available
                for metric_key, metric_data in perf_metrics.items():
                    if isinstance(metric_data, dict) and metric_data.get('success', False):
                        enhanced_point = data_point.copy()
                        enhanced_point.update({
                            'image_size_pixels': metric_data.get('image_size_pixels', 1),
                            'inference_time': metric_data.get('inference_time', execution_time),
                            'throughput': metric_data.get('throughput_px_per_sec', 0),
                            'precision': metric_data.get('precision_used', 'unknown'),
                            'resolution': metric_data.get('resolution', 'unknown')
                        })
                        plot_data.append(enhanced_point)
            else:
                # Simple benchmark data
                plot_data.append(data_point)
        
        logger.debug(f"Extracted {len(plot_data)} data points for plotting")
        return plot_data
    
    def _create_plot_config(self, plot_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create plot configuration based on the data."""
        config = {
            'figsize': (12, 8),
            'title': 'Performance Comparison',
            'xlabel': 'X Axis',
            'ylabel': 'Y Axis',
            'xscale': 'linear',
            'yscale': 'linear',
            'grid': True,
            'legend_location': 'best'
        }
        
        # Extract custom title from data if available
        for point in plot_data:
            if 'benchmark_title' in point:
                config['title'] = point['benchmark_title']
                break
        
        # Determine plot type and axes based on available data
        has_complex_data = any('image_size_pixels' in point for point in plot_data)
        has_batch_data = any(point.get('batch_sizes') and point.get('times') for point in plot_data)
        
        if has_complex_data:
            # Complex benchmark data - inference time vs image size
            config.update({
                'xlabel': 'Image Size (pixels)',
                'ylabel': 'Inference Time (seconds)',
                'xscale': 'log',
                'yscale': 'log',
                'plot_type': 'complex'
            })
        elif has_batch_data:
            # Batch size vs time data
            config.update({
                'xlabel': 'Batch Size',
                'ylabel': 'Time (seconds)',
                'xscale': 'log',
                'yscale': 'log',
                'plot_type': 'batch'
            })
        else:
            # Simple execution time comparison
            config.update({
                'xlabel': 'Device',
                'ylabel': 'Execution Time (seconds)',
                'xscale': 'linear',
                'yscale': 'linear',
                'plot_type': 'simple'
            })
        
        return config
    
    def _extract_vendor(self, flavour: str, device_name: str) -> str:
        """Extract vendor from flavour or device name."""
        flavour_lower = flavour.lower()
        device_name_lower = device_name.lower()
        
        if 'nvidia' in flavour_lower or 'cuda' in device_name_lower:
            return 'nvidia'
        elif 'intel' in flavour_lower or 'xpu' in device_name_lower:
            return 'intel'
        elif 'amd' in flavour_lower or 'hip' in device_name_lower or 'rocm' in device_name_lower:
            return 'amd'
        else:
            return 'unknown'
    
    def _group_data_by_device(self, plot_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group data points by device for plotting."""
        device_data = {}
        
        for point in plot_data:
            # Use device_model as the key if available and meaningful, otherwise fall back to vendor_type
            device_model = point.get('device_model', '')
            vendor = point.get('vendor', 'unknown')
            device_type = point.get('device_type', 'unknown')
            device_name = point.get('device_name', 'Unknown')
            
            # Determine the best device key for grouping
            if device_model and device_model != device_name and device_model.lower() != 'unknown':
                # Use device model if it's more specific than device name
                device_key = device_model
            elif device_name and device_name.lower() != 'unknown':
                # Use device name if it's meaningful
                device_key = device_name
            else:
                # Fall back to vendor_type grouping
                device_key = f"{vendor}_{device_type}"
                
            if device_key not in device_data:
                device_data[device_key] = []
            device_data[device_key].append(point)
        
        return device_data
    
    def _group_data_by_flavour(self, plot_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Group data points by flavour, then by device within each flavour."""
        flavour_data = {}
        
        for point in plot_data:
            # Extract flavour information
            vendor = point.get('vendor', 'unknown')
            flavour_name = point.get('flavour_name', f"{vendor.title()}")
            device_type = point.get('device_type', 'unknown')
            device_model = point.get('device_model', '')
            device_name = point.get('device_name', 'Unknown')
            
            # Create flavour key
            flavour_key = vendor
            
            # Create device key within flavour
            if device_model and device_model != device_name and device_model.lower() != 'unknown':
                device_key = f"{device_model} ({device_type.upper()})"
            elif device_name and device_name.lower() != 'unknown':
                device_key = f"{device_name} ({device_type.upper()})"
            else:
                device_key = f"{flavour_name} {device_type.upper()}"
            
            # Initialize flavour group if needed
            if flavour_key not in flavour_data:
                flavour_data[flavour_key] = {}
            
            # Initialize device group within flavour if needed
            if device_key not in flavour_data[flavour_key]:
                flavour_data[flavour_key][device_key] = []
            
            flavour_data[flavour_key][device_key].append(point)
        
        logger.info(f"Grouped data by flavour: {list(flavour_data.keys())}")
        return flavour_data
    
    def _plot_flavour_performance(self, ax, flavour_key: str, flavour_devices: Dict[str, List[Dict[str, Any]]], plot_config: Dict[str, Any]):
        """Plot performance data for all devices in a flavour."""
        # Get flavour color
        color = self.flavour_colors.get(flavour_key, self.flavour_colors['unknown'])
        
        plot_type = plot_config.get('plot_type', 'simple')
        
        # Plot each device in this flavour
        for device_key, device_metrics in flavour_devices.items():
            if not device_metrics:
                continue
                
            # Extract device type for styling
            first_point = device_metrics[0]
            device_type = first_point.get('device_type', 'unknown')
            style = self.device_styles.get(device_type, self.device_styles['cpu'])
            
            # Create label for this device
            flavour_name = first_point.get('flavour_name', flavour_key.title())
            label = f"{flavour_name}: {device_key}"
            
            if plot_type == 'complex':
                self._plot_flavour_complex_metrics(ax, device_key, device_metrics, color, style, label)
            elif plot_type == 'batch':
                self._plot_flavour_batch_metrics(ax, device_key, device_metrics, color, style, label)
            else:
                self._plot_flavour_simple_metrics(ax, device_key, device_metrics, color, style, label)
    
    def _plot_flavour_complex_metrics(self, ax, device_key: str, device_metrics: List[Dict[str, Any]], 
                                    color: str, style: Dict[str, Any], label: str):
        """Plot complex benchmark metrics for a device within a flavour."""
        # Extract data
        image_sizes = [point['image_size_pixels'] for point in device_metrics if 'image_size_pixels' in point]
        inference_times = [point['inference_time'] for point in device_metrics if 'inference_time' in point]
        resolutions = [point['resolution'] for point in device_metrics if 'resolution' in point]
        
        if len(image_sizes) < 1 or len(inference_times) < 1:
            return
        
        # Convert to numpy arrays
        pixels = np.array(image_sizes)
        times = np.array(inference_times)
        
        # Sort by pixels for proper plotting
        sort_idx = np.argsort(pixels)
        pixels_sorted = pixels[sort_idx]
        times_sorted = times[sort_idx]
        
        # Collect resolution mapping for secondary x-axis
        if not hasattr(ax, '_resolution_mapping'):
            ax._resolution_mapping = {}
        for i, pixel_count in enumerate(pixels):
            if i < len(resolutions):
                ax._resolution_mapping[pixel_count] = resolutions[i]
        
        # Plot line if we have multiple points
        if len(pixels_sorted) >= 2:
            ax.plot(pixels_sorted, times_sorted,
                   color=color,
                   linestyle=style['linestyle'],
                   linewidth=2.5 if 'gpu' in device_key.lower() else 2.0,
                   label=label,
                   alpha=0.8)
        
        # Plot data points
        ax.scatter(pixels_sorted, times_sorted,
                  color=color,
                  marker=style['marker'],
                  s=style['markersize']**2,
                  alpha=0.6,
                  zorder=5,
                  edgecolors='white',
                  linewidth=0.5)
    
    def _plot_flavour_batch_metrics(self, ax, device_key: str, device_metrics: List[Dict[str, Any]], 
                                  color: str, style: Dict[str, Any], label: str):
        """Plot batch size vs time metrics for a device within a flavour."""
        # Extract batch data from the first point that has it
        batch_data = None
        for point in device_metrics:
            if point.get('batch_sizes') and point.get('times'):
                batch_data = point
                break
        
        if not batch_data:
            return
        
        batch_sizes = batch_data['batch_sizes']
        times = batch_data['times']
        
        if len(batch_sizes) != len(times) or len(batch_sizes) < 1:
            return
        
        # Plot the data
        ax.plot(batch_sizes, times,
               color=color,
               linestyle=style['linestyle'],
               marker=style['marker'],
               markersize=style['markersize'],
               label=label,
               alpha=0.8)
    
    def _plot_flavour_simple_metrics(self, ax, device_key: str, device_metrics: List[Dict[str, Any]], 
                                    color: str, style: Dict[str, Any], label: str):
        """Plot simple benchmark metrics for a device within a flavour."""
        # For simple metrics, we'll plot execution time
        execution_times = [point['execution_time'] for point in device_metrics]
        device_indices = list(range(len(execution_times)))
        
        if not execution_times:
            return
        
        # Plot as bar chart for simple metrics
        ax.bar(device_indices, execution_times,
               color=color,
               alpha=0.7,
               label=label,
               width=0.6)
        
        # Add text labels on bars
        for i, time_val in enumerate(execution_times):
            ax.text(i, time_val + max(execution_times) * 0.01,
                   f'{time_val:.3f}s',
                   ha='center', va='bottom', fontsize=10)
    
    def _customize_flavour_plot(self, ax, flavour_data: Dict[str, Dict[str, List[Dict[str, Any]]]], plot_config: Dict[str, Any]):
        """Customize the plot appearance for flavour-based plotting."""
        # Apply configuration from data
        ax.set_title(plot_config['title'], fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(plot_config['xlabel'], fontsize=12, fontweight='bold')
        ax.set_ylabel(plot_config['ylabel'], fontsize=12, fontweight='bold')
        ax.set_xscale(plot_config['xscale'])
        ax.set_yscale(plot_config['yscale'])
        
        # Add grid if configured
        if plot_config.get('grid', True):
            ax.grid(True, alpha=0.3)
        
        # Handle special case for complex data with resolution mapping
        if plot_config.get('plot_type') == 'complex' and hasattr(ax, '_resolution_mapping') and ax._resolution_mapping:
            # Add secondary x-axis at top with resolution labels
            ax2 = ax.twiny()
            
            # Get the pixel values and their corresponding resolutions
            pixel_values = sorted(ax._resolution_mapping.keys())
            resolution_labels = [ax._resolution_mapping[pixels] for pixels in pixel_values]
            
            # Set the secondary axis to have the same scale as primary
            ax2.set_xscale(plot_config['xscale'])
            ax2.set_xlim(ax.get_xlim())
            
            # Set tick positions and labels
            ax2.set_xticks(pixel_values)
            ax2.set_xticklabels(resolution_labels, fontsize=10)
            ax2.set_xlabel('Resolution (width × height)', fontsize=12, fontweight='bold')
            
            # Style the secondary axis
            ax2.tick_params(axis='x', which='major', labelsize=10)
        
        # Handle simple plots with device names on x-axis
        if plot_config.get('plot_type') == 'simple':
            device_names = []
            for flavour_devices in flavour_data.values():
                for device_key in flavour_devices.keys():
                    device_names.append(device_key)
            
            if device_names:
                ax.set_xticks(range(len(device_names)))
                ax.set_xticklabels(device_names, rotation=45, ha='right')
        
        # Add custom legend with flavour grouping
        self._add_flavour_grouped_legend(ax, flavour_data, plot_config)
    
    def _add_flavour_grouped_legend(self, ax, flavour_data: Dict[str, Dict[str, List[Dict[str, Any]]]], plot_config: Dict[str, Any]):
        """Add a custom legend that groups devices under their flavour headings."""
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        
        # Get legend location from config
        legend_location = plot_config.get('legend_location', 'best')
        
        # Collect legend elements
        legend_elements = []
        
        # Iterate through flavours in a consistent order
        sorted_flavours = sorted(flavour_data.keys())
        
        for flavour_key in sorted_flavours:
            flavour_devices = flavour_data[flavour_key]
            
            if not flavour_devices:
                continue
            
            # Get flavour info from first device
            first_device_data = next(iter(flavour_devices.values()))[0]
            flavour_name = first_device_data.get('flavour_name', flavour_key.title())
            
            # Add flavour header as a text-only legend entry
            flavour_header = Line2D([0], [0], color='none', label=f'Flavour {flavour_name}')
            legend_elements.append(flavour_header)
            
            # Get flavour color
            flavour_color = self.flavour_colors.get(flavour_key, self.flavour_colors['unknown'])
            
            # Add devices under this flavour with CPU first, then GPU
            sorted_devices = self._sort_devices_by_type(flavour_devices.keys(), flavour_devices)
            for device_key in sorted_devices:
                device_data_list = flavour_devices[device_key]
                if not device_data_list:
                    continue
                
                # Get device styling
                first_point = device_data_list[0]
                device_type = first_point.get('device_type', 'unknown')
                style = self.device_styles.get(device_type, self.device_styles['cpu'])
                
                # Create device legend entry with appropriate marker
                device_line = Line2D([0], [0], 
                                   color=flavour_color,
                                   marker=style['marker'],
                                   linestyle=style['linestyle'],
                                   markersize=8,
                                   linewidth=2.0,
                                   label=f"  {device_key}")  # Indent device names
                legend_elements.append(device_line)
        
        # Create the legend
        if legend_location == 'outside':
            legend = ax.legend(handles=legend_elements, 
                             bbox_to_anchor=(1.05, 1), 
                             loc='upper left', 
                             frameon=True, 
                             fancybox=True, 
                             shadow=True,
                             fontsize=10)
        else:
            legend = ax.legend(handles=legend_elements, 
                             loc=legend_location, 
                             frameon=True, 
                             fancybox=True, 
                             shadow=True,
                             fontsize=10)
        
        # Customize legend appearance
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # Make flavour headers bold and remove their markers
        # Handle both old and new matplotlib API
        try:
            legend_handles = legend.legend_handles  # New matplotlib API
        except AttributeError:
            legend_handles = legend.legendHandles   # Old matplotlib API
        
        for i, (text, handle) in enumerate(zip(legend.get_texts(), legend_handles)):
            label = text.get_text()
            if label.startswith('Flavour '):
                # Make flavour headers bold and hide their line
                text.set_weight('bold')
                text.set_fontsize(11)
                handle.set_visible(False)
            elif label.startswith('  '):
                # Device entries - ensure proper styling
                text.set_fontsize(9)
    
    def _sort_devices_by_type(self, device_keys, flavour_devices):
        """Sort devices with CPU first (dotted lines), then GPU (solid lines)."""
        cpu_devices = []
        gpu_devices = []
        
        for device_key in device_keys:
            device_data_list = flavour_devices[device_key]
            if device_data_list:
                device_type = device_data_list[0].get('device_type', 'unknown').lower()
                if device_type == 'cpu':
                    cpu_devices.append(device_key)
                else:
                    # All non-CPU devices (gpu, xpu, cuda, hip, rocm, etc.) go to GPU list
                    gpu_devices.append(device_key)
        
        # Sort within each category and return CPU devices first, then GPU devices
        return sorted(cpu_devices) + sorted(gpu_devices)
    
    def _plot_device_performance(self, ax, device_key: str, device_metrics: List[Dict[str, Any]], plot_config: Dict[str, Any]):
        """Plot performance data for a single device."""
        # Extract vendor and device_type from the data instead of parsing the key
        first_point = device_metrics[0]
        vendor = first_point.get('vendor', 'unknown')
        device_type = first_point.get('device_type', 'unknown')
        
        # Get styling
        color = self.vendor_colors.get(vendor, self.vendor_colors['unknown'])
        style = self.device_styles.get(device_type, self.device_styles['cpu'])
        
        plot_type = plot_config.get('plot_type', 'simple')
        
        if plot_type == 'complex':
            self._plot_complex_metrics(ax, device_key, device_metrics, color, style)
        elif plot_type == 'batch':
            self._plot_batch_metrics(ax, device_key, device_metrics, color, style)
        else:
            self._plot_simple_metrics(ax, device_key, device_metrics, color, style)
    
    def _plot_complex_metrics(self, ax, device_key: str, device_metrics: List[Dict[str, Any]], 
                            color: str, style: Dict[str, Any]):
        """Plot complex benchmark metrics (inference time vs image size)."""
        # Extract data
        image_sizes = [point['image_size_pixels'] for point in device_metrics if 'image_size_pixels' in point]
        inference_times = [point['inference_time'] for point in device_metrics if 'inference_time' in point]
        resolutions = [point['resolution'] for point in device_metrics if 'resolution' in point]
        
        if len(image_sizes) < 1 or len(inference_times) < 1:
            return
        
        # Get device information from the data - prioritize device_model for legend
        device_model = device_metrics[0].get('device_model', '')
        device_name = device_metrics[0].get('device_name', 'Unknown')
        flavour_name = device_metrics[0].get('flavour_name', device_key)
        
        # Determine label based on available data
        if device_model and device_model != device_name and device_model.lower() != 'unknown':
            label = device_model  # Use device model if it's more specific
        elif device_name and device_name.lower() != 'unknown':
            label = device_name   # Use device name if available
        else:
            label = flavour_name  # Fall back to flavour name
        
        # Convert to numpy arrays
        pixels = np.array(image_sizes)
        times = np.array(inference_times)
        
        # Sort by pixels for proper plotting
        sort_idx = np.argsort(pixels)
        pixels_sorted = pixels[sort_idx]
        times_sorted = times[sort_idx]
        
        # Collect resolution mapping for secondary x-axis
        if not hasattr(ax, '_resolution_mapping'):
            ax._resolution_mapping = {}
        for i, pixel_count in enumerate(pixels):
            if i < len(resolutions):
                ax._resolution_mapping[pixel_count] = resolutions[i]
        
        # Plot line if we have multiple points
        if len(pixels_sorted) >= 2:
            ax.plot(pixels_sorted, times_sorted,
                   color=color,
                   linestyle=style['linestyle'],
                   linewidth=2.5 if 'gpu' in device_key.lower() else 2.0,
                   label=label,
                   alpha=0.8)
        
        # Plot data points
        ax.scatter(pixels_sorted, times_sorted,
                  color=color,
                  marker=style['marker'],
                  s=style['markersize']**2,
                  alpha=0.6,
                  zorder=5,
                  edgecolors='white',
                  linewidth=0.5)
    
    def _plot_batch_metrics(self, ax, device_key: str, device_metrics: List[Dict[str, Any]], 
                          color: str, style: Dict[str, Any]):
        """Plot batch size vs time metrics."""
        # Extract batch data from the first point that has it
        batch_data = None
        for point in device_metrics:
            if point.get('batch_sizes') and point.get('times'):
                batch_data = point
                break
        
        if not batch_data:
            return
        
        batch_sizes = batch_data['batch_sizes']
        times = batch_data['times']
        
        if len(batch_sizes) != len(times) or len(batch_sizes) < 1:
            return
        
        # Get device information for legend
        device_model = batch_data.get('device_model', '')
        device_name = batch_data.get('device_name', 'Unknown')
        flavour_name = batch_data.get('flavour_name', device_key)
        
        # Determine label
        if device_model and device_model != device_name and device_model.lower() != 'unknown':
            label = device_model
        elif device_name and device_name.lower() != 'unknown':
            label = device_name
        else:
            label = flavour_name
        
        # Plot the data
        ax.plot(batch_sizes, times,
               color=color,
               linestyle=style['linestyle'],
               marker=style['marker'],
               markersize=style['markersize'],
               label=label,
               alpha=0.8)
    
    def _plot_simple_metrics(self, ax, device_key: str, device_metrics: List[Dict[str, Any]], 
                            color: str, style: Dict[str, Any]):
        """Plot simple benchmark metrics (execution time vs device)."""
        # For simple metrics, we'll plot execution time
        execution_times = [point['execution_time'] for point in device_metrics]
        device_indices = list(range(len(execution_times)))
        
        if not execution_times:
            return
        
        # Get device information from the data - prioritize device_model for legend
        device_model = device_metrics[0].get('device_model', '')
        device_name = device_metrics[0].get('device_name', 'Unknown')
        flavour_name = device_metrics[0].get('flavour_name', device_key)
        
        # Determine label based on available data
        if device_model and device_model != device_name and device_model.lower() != 'unknown':
            label = device_model  # Use device model if it's more specific
        elif device_name and device_name.lower() != 'unknown':
            label = device_name   # Use device name if available
        else:
            label = flavour_name  # Fall back to flavour name
        
        # Plot as bar chart for simple metrics
        ax.bar(device_indices, execution_times,
               color=color,
               alpha=0.7,
               label=label,
               width=0.6)
        
        # Add text labels on bars
        for i, time_val in enumerate(execution_times):
            ax.text(i, time_val + max(execution_times) * 0.01,
                   f'{time_val:.3f}s',
                   ha='center', va='bottom', fontsize=10)
    
    def _customize_plot(self, ax, device_data: Dict[str, List[Dict[str, Any]]], plot_config: Dict[str, Any]):
        """Customize the plot appearance based on configuration."""
        # Apply configuration from data
        ax.set_title(plot_config['title'], fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(plot_config['xlabel'], fontsize=12, fontweight='bold')
        ax.set_ylabel(plot_config['ylabel'], fontsize=12, fontweight='bold')
        ax.set_xscale(plot_config['xscale'])
        ax.set_yscale(plot_config['yscale'])
        
        # Add grid if configured
        if plot_config.get('grid', True):
            ax.grid(True, alpha=0.3)
        
        # Handle special case for complex data with resolution mapping
        if plot_config.get('plot_type') == 'complex' and hasattr(ax, '_resolution_mapping') and ax._resolution_mapping:
            # Add secondary x-axis at top with resolution labels
            ax2 = ax.twiny()
            
            # Get the pixel values and their corresponding resolutions
            pixel_values = sorted(ax._resolution_mapping.keys())
            resolution_labels = [ax._resolution_mapping[pixels] for pixels in pixel_values]
            
            # Set the secondary axis to have the same scale as primary
            ax2.set_xscale(plot_config['xscale'])
            ax2.set_xlim(ax.get_xlim())
            
            # Set tick positions and labels
            ax2.set_xticks(pixel_values)
            ax2.set_xticklabels(resolution_labels, fontsize=10)
            ax2.set_xlabel('Resolution (width × height)', fontsize=12, fontweight='bold')
            
            # Style the secondary axis
            ax2.tick_params(axis='x', which='major', labelsize=10)
        
        # Handle simple plots with device names on x-axis
        if plot_config.get('plot_type') == 'simple':
            device_names = []
            for metrics in device_data.values():
                for point in metrics:
                    device_names.append(point['device_name'])
            
            if device_names:
                ax.set_xticks(range(len(device_names)))
                ax.set_xticklabels(device_names, rotation=45, ha='right')
        
        # Add legend
        legend_location = plot_config.get('legend_location', 'best')
        if legend_location == 'outside':
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        else:
            ax.legend(loc=legend_location, frameon=True, fancybox=True, shadow=True)
    
    def store_plot(self, filepath: str):
        """
        Store the plot(s) to file(s).
        
        Args:
            filepath: Path where to save the plot (supports .png, .svg, .pdf)
                     For multiple plots, precision will be inserted into filename
        """
        if self.plot is None:
            logger.error("No plot available to store")
            return
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Handle multiple plots (dict of precision -> figure)
            if isinstance(self.plot, dict):
                base_path = filepath.with_suffix('')
                for precision, fig in self.plot.items():
                    # Create precision-specific filename
                    precision_path = base_path.with_name(f"{base_path.name}_{precision}")
                    
                    # Save as PNG
                    png_path = precision_path.with_suffix('.png')
                    fig.savefig(png_path, dpi=300, bbox_inches='tight')
                    logger.debug(f"Plot saved as PNG: {png_path}")
                    
                    # Save as SVG for vector graphics
                    svg_path = precision_path.with_suffix('.svg')
                    fig.savefig(svg_path, bbox_inches='tight')
                    logger.debug(f"Plot saved as SVG: {svg_path}")
                
                logger.info(f"Saved {len(self.plot)} precision plots to {filepath.parent}")
            
            # Handle single plot (figure)
            else:
                # Save in multiple formats for compatibility
                base_path = filepath.with_suffix('')
                
                # Always save as PNG
                png_path = base_path.with_suffix('.png')
                self.plot.savefig(png_path, dpi=300, bbox_inches='tight')
                logger.debug(f"Plot saved as PNG: {png_path}")
                
                # Also save as SVG for vector graphics
                svg_path = base_path.with_suffix('.svg')
                self.plot.savefig(svg_path, bbox_inches='tight')
                logger.debug(f"Plot saved as SVG: {svg_path}")
                
                logger.info(f"Plot saved successfully to {filepath.parent}")
            
            plt.close('all')  # Clean up all figures
            
        except Exception as e:
            logger.error(f"Failed to save plot to {filepath}: {e}")
            raise

class ImageDifferencePlotter:
    """
    A class for plotting image differences.
    """
    def __init__(self, data):
        self.data = data

    def plot(self):
        # Implementation for plotting image differences
        pass