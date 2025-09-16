"""
The Benchmark Manager is the entry point for the benchmarking process.
It orchestrates the loading, running, and evaluation of benchmarks.
"""

import logging
from typing import Any, Dict, List, Optional
import os

from ..benchmarks.base import BaseBenchmark

logger = logging.getLogger(__name__)

class BenchmarkManager:
    """
    Manages the benchmarking process, including loading and running benchmarks.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initialize the BenchmarkManager with configuration and managers."""
        self.config = config
        flavour = os.getenv('ARKITEKT_FLAVOUR', None)
        self.config['arkitekt_flavour'] = flavour

        self.benchmarks = self._load_benchmarks()
        self.run()

    def _load_benchmarks(self) -> List[BaseBenchmark]:
        """Load benchmark classes based on configuration."""
        enabled_benchmarks = self.config.get('benchmarks', {}).get('enabled', [])
        benchmark_params = self.config.get('benchmarks', {}).get('parameters', {})
        benchmarks = []
        
        for benchmark_name in enabled_benchmarks:
            logger.debug(f"Loading benchmark: {benchmark_name}")
            # Create full config for benchmark (global config + benchmark-specific params)
            benchmark_specific_params = benchmark_params.get(benchmark_name, {})
            full_config = self.config.copy()  # Start with global config
            full_config.update(benchmark_specific_params)  # Add/override with benchmark-specific params
            
            # Load benchmark class dynamically and pass the full configuration
            benchmark = self._create_benchmark(benchmark_name, full_config)
            if benchmark:
                benchmarks.append(benchmark)
                logger.debug(f"Successfully loaded benchmark: {benchmark_name}")
        logger.info(f"Total benchmarks loaded: {len(benchmarks)}")
        for benchmark in benchmarks:
            logger.info(f" - {benchmark.name}")
        return benchmarks
    
    def _create_benchmark(self, name: str, params: Dict[str, Any]) -> Optional[BaseBenchmark]:
        """Create a benchmark instance by name dynamically."""
        try:
            # Convert benchmark name to module name (e.g., "test_benchmark" -> "test_benchmark")
            module_name = name
            
            # Import the benchmark module dynamically
            from importlib import import_module
            benchmark_module = import_module(f"bench_ark.benchmarks.{module_name}")
            
            # Find the benchmark class in the module
            # Convention: Class name should be CamelCase version of module name
            # e.g., "test_benchmark" -> "TestBenchmark"
            class_name = self._module_name_to_class_name(module_name)
            
            if hasattr(benchmark_module, class_name):
                benchmark_class = getattr(benchmark_module, class_name)
                return benchmark_class(name, params)
            else:
                # Try to find any class that inherits from BaseBenchmark
                for attr_name in dir(benchmark_module):
                    attr = getattr(benchmark_module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseBenchmark) and 
                        attr != BaseBenchmark):
                        logger.info(f"Found benchmark class '{attr_name}' in module '{module_name}'")
                        return attr(name, params)
                
                logger.error(f"No valid benchmark class found in module '{module_name}'")
                return None
                
        except ImportError as e:
            logger.error(f"Failed to import benchmark module '{name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create benchmark '{name}': {e}")
            return None
    
    def _module_name_to_class_name(self, module_name: str) -> str:
        """Convert module name to expected class name using CamelCase convention."""
        # Split by underscore and capitalize each part
        parts = module_name.split('_')
        return ''.join(word.capitalize() for word in parts)

    def run(self) -> None:
        """
        Running the benchmarks
        """
        if self.config.get('run_benchmarks', False):
            self._run_benchmarks()
        
        if self.config.get('evaluate_benchmarks', False):
            self._evaluate_benchmarks()

    def _run_benchmarks(self) -> None:
        """Run all loaded benchmarks and return results."""
        if not self.benchmarks:
            logger.error("No benchmarks loaded to run")
            return {}

        for benchmark in self.benchmarks:
            benchmark.run()

    def _evaluate_benchmarks(self) -> None:
        """Evaluate all loaded benchmarks and return results."""
        if not self.benchmarks:
            logger.error("No benchmarks loaded to evaluate")
            return {}

        for benchmark in self.benchmarks:
            benchmark.evaluate()
        