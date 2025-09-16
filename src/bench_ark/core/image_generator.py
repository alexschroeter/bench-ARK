"""
Image generation utilities for artificial benchmarking data.
"""

import logging
import time
import hashlib
import numpy as np
from typing import Tuple, Dict, Any
import multiprocessing as mp

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generates artificial test images with ground truth for benchmarking."""
    
    @staticmethod
    def generate_artificial_image_adaptive(resolution: Tuple[int, int], benchmark_name: str = "default") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adaptive image generation that automatically chooses the best method based on image size.
        
        Args:
            resolution: Image resolution (height, width)
            benchmark_name: Name for deterministic generation
            
        Returns:
            tuple: (image, ground_truth_data)
        """
        height, width = resolution
        total_pixels = height * width
        
        # Automatic method selection based on image size
        if total_pixels >= 250_000:  # >= 250K pixels (500x500)
            logger.info(f"Large image detected ({total_pixels:,} pixels) - using multicore generation")
            return ImageGenerator.generate_artificial_image_multicore(resolution, benchmark_name)
        else:
            logger.info(f"Small image ({total_pixels:,} pixels) - using single-threaded generation")
            return ImageGenerator.generate_artificial_image_with_ground_truth(resolution, benchmark_name)
    
    @staticmethod
    def generate_artificial_image_with_ground_truth(resolution: Tuple[int, int], benchmark_name: str = "default") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate deterministic artificial test image with cell-like structures and ground truth.
        
        Args:
            resolution: Image resolution (height, width)
            benchmark_name: Name for deterministic generation
            
        Returns:
            tuple: (image, ground_truth_data) where ground_truth_data contains:
                - cells: list of cell dictionaries
                - ground_truth_mask: 2D array with cell IDs
                - metadata: dict with generation parameters
        """
        # Create deterministic seed from benchmark name and resolution
        seed_string = f"{benchmark_name}_{resolution[0]}x{resolution[1]}"
        seed_hash = hashlib.md5(seed_string.encode('utf-8')).hexdigest()
        seed = int(seed_hash[:8], 16)
        np.random.seed(seed)
        
        height, width = resolution
        image = np.zeros((height, width, 3), dtype=np.uint8)
        ground_truth_mask = np.zeros((height, width), dtype=np.uint16)
        
        # Store cell information
        cells_data = []
        
        # Create background with deterministic noise
        background = np.random.normal(50, 10, (height, width))
        background = np.clip(background, 0, 255).astype(np.uint8)
        
        # Add cell-like circular structures
        base_cells = (height * width) // 10000
        num_cells = max(5, min(base_cells, 1000))  # Cap at 1000 for performance
        
        logger.info(f"Generating {num_cells} cells for {width}x{height} image...")
        
        y, x = np.ogrid[:height, :width]
        
        for cell_id in range(1, num_cells + 1):  # Start from 1 (0 is background)
            center_x = np.random.randint(20, width - 20)
            center_y = np.random.randint(20, height - 20)
            radius = np.random.randint(8, 25)
            intensity = np.random.randint(100, 200)
            
            # Create circular mask for this cell
            current_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Store cell data
            cell_data = {
                'cell_id': cell_id,
                'center_x': int(center_x),
                'center_y': int(center_y),
                'radius': int(radius),
                'intensity': int(intensity),
                'area_pixels': int(np.sum(current_mask))
            }
            cells_data.append(cell_data)
            
            # Add to ground truth mask
            ground_truth_mask[current_mask] = cell_id
            
            # Add cell to all channels
            cell_noise = np.random.normal(0, 5, np.sum(current_mask)).astype(int)
            for c in range(3):
                image[current_mask, c] = intensity + cell_noise
        
        # Add background to all channels
        for c in range(3):
            image[:, :, c] = np.maximum(image[:, :, c], background)
        
        # Create ground truth data structure
        ground_truth_data = {
            'cells': cells_data,
            'ground_truth_mask': ground_truth_mask,
            'metadata': {
                'resolution': resolution,
                'benchmark_name': benchmark_name,
                'seed': seed,
                'num_cells': len(cells_data),
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Reset random seed
        np.random.seed(None)
        
        return np.clip(image, 0, 255).astype(np.uint8), ground_truth_data
    
    @staticmethod 
    def generate_artificial_image_multicore(resolution: Tuple[int, int], benchmark_name: str = "default") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fast multi-core image generation for larger images.
        
        Args:
            resolution: Image resolution (height, width)
            benchmark_name: Name for deterministic generation
            
        Returns:
            tuple: (image, ground_truth_data)
        """
        height, width = resolution
        total_pixels = height * width
        
        # Create deterministic seed
        seed_string = f"{benchmark_name}_{resolution[0]}x{resolution[1]}"
        seed_hash = hashlib.md5(seed_string.encode('utf-8')).hexdigest()
        base_seed = int(seed_hash[:8], 16)
        np.random.seed(base_seed)
        
        # Calculate optimal parameters
        max_cells = min(1000, max(5, total_pixels // 15000))
        max_workers = min(8, mp.cpu_count())
        
        logger.info(f"Multi-core generation: {max_cells} cells for {width}x{height} image using {max_workers} cores")
        
        start_time = time.time()
        
        # Pre-generate cell parameters
        centers_x = np.random.randint(20, width - 20, max_cells)
        centers_y = np.random.randint(20, height - 20, max_cells)
        radii = np.random.randint(8, 25, max_cells)
        intensities = np.random.randint(100, 200, max_cells)
        
        # Create image and mask
        image = np.zeros((height, width, 3), dtype=np.uint8)
        ground_truth_mask = np.zeros((height, width), dtype=np.uint16)
        
        # Generate background
        np.random.seed(base_seed)  # Reset for consistent background
        background = np.random.normal(50, 10, (height, width))
        background = np.clip(background, 0, 255).astype(np.uint8)
        
        # Apply background to all channels
        for c in range(3):
            image[:, :, c] = background
        
        # Add cells
        y, x = np.ogrid[:height, :width]
        cells_data = []
        
        for cell_id in range(1, max_cells + 1):
            center_x = centers_x[cell_id - 1]
            center_y = centers_y[cell_id - 1]
            radius = radii[cell_id - 1]
            intensity = intensities[cell_id - 1]
            
            # Create circular mask
            current_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Store cell data
            cells_data.append({
                'cell_id': cell_id,
                'center_x': int(center_x),
                'center_y': int(center_y),
                'radius': int(radius),
                'intensity': int(intensity),
                'area_pixels': int(np.sum(current_mask))
            })
            
            # Add to mask and image
            ground_truth_mask[current_mask] = cell_id
            
            # Add noise and apply to all channels
            np.random.seed(base_seed + cell_id)
            cell_noise = np.random.normal(0, 5, np.sum(current_mask)).astype(int)
            for c in range(3):
                image[current_mask, c] = intensity + cell_noise
        
        total_time = time.time() - start_time
        logger.info(f"Generation completed in {total_time:.3f}s ({max_cells/total_time:.1f} cells/sec)")
        
        # Create ground truth data
        ground_truth_data = {
            'cells': cells_data,
            'ground_truth_mask': ground_truth_mask,
            'metadata': {
                'num_cells': max_cells,
                'resolution': resolution,
                'generation_method': 'multicore_optimized',
                'generation_time_seconds': total_time,
                'cells_per_second': max_cells / total_time if total_time > 0 else 0,
                'benchmark_name': benchmark_name,
                'total_pixels': total_pixels
            }
        }
        
        np.random.seed(None)
        return np.clip(image, 0, 255).astype(np.uint8), ground_truth_data
