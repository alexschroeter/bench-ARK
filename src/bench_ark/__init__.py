"""
bench-ARK: A comprehensive benchmarking framework for ARK (AI Recognition Kit) models.

This package provides tools for benchmarking AI models with comprehensive
device support, performance metrics, and result analysis.
"""

__version__ = "0.1.0"
__author__ = "ARK Team"
__email__ = "team@ark.ai"

from .core.benchmark_manager import BenchmarkManager
from .core.device_manager import DeviceManager

__all__ = [
    "BenchmarkManager",
    "DeviceManager", 
]
