from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import json
import os
from datetime import datetime
from pathlib import Path

from ..core.device_manager import DeviceManager

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """
    This is the Abstract Base Class for all Benchmarks
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the benchmark with the given configuration.
        """
        self.name = name
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_results = []
        self.all_results = []
        self.device_manager = DeviceManager()
        self.device_manager._set_available_devices(self._get_available_devices())
        logger.info(f"Benchmark {self.name} found {len(self.device_manager._list_available_devices())} available devices.")
        for device in self.device_manager._list_available_devices():
            device_type = getattr(device, "type", "unknow")
            logger.info(f" - {device.name} ({device_type})")

    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark.
        """
        self.current_results = self._run_benchmark()
        self._store_current_benchmark_results()


    def evaluate(self):
        self.all_results = self._load_benchmark_results()
        self._evaluate_benchmark()

    @abstractmethod
    def _get_available_devices(self) -> List[Dict[str, Any]]:
        """Get a list of available devices for benchmarking."""
        return []

    @abstractmethod
    def _run_benchmark(self) -> Dict[str, Any]:
        return {}

    def _store_current_benchmark_results(self) -> None:
        """
        Store the results of the benchmark in a JSON file.
        Path: <current directory>/<dataset from config>/<benchmark name>/<timestamp>_<flavour from config>/results.json
        """
        try:
            # Get dataset from config (default to 'default_dataset' if not specified)
            dataset = self.config.get('dataset', 'default_dataset')
            
            # Get flavour from environment variable (as implemented in device_manager)
            flavour = self.config.get('arkitekt_flavour', 'vanilla')

            # Create timestamp (use existing one if set by benchmark, otherwise create new)
            timestamp = getattr(self, 'timestamp', None) or datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create directory structure: <current_dir>/<dataset>/<benchmark_name>/<timestamp>_<flavour>/
            current_dir = Path.cwd()
            results_dir = current_dir / dataset / self.name / f"{timestamp}_{flavour}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create results file path
            results_file = results_dir / "results.json"
            
            # Get device information from device manager
            devices_info = []
            for device in self.device_manager._list_available_devices():
                # Convert Device object to dictionary for JSON serialization
                devices_info.append(device.__dict__)
            
            # Prepare results data for JSON serialization
            results_data = {
                "benchmark_name": self.name,
                "dataset": dataset,
                "flavour": flavour,
                "timestamp": timestamp,
                "devices": devices_info,
                "config": self.config,
                "results": self.current_results
            }
            
            # Write results to JSON file
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Benchmark results stored in: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to store benchmark results: {e}")
            raise

    def _load_benchmark_results(self) -> List[Dict[str, Any]]:
        """
        Load benchmark results from the dataset folder.
        
        By default loads the latest results for each flavour found in:
        <config.dataset>/<benchmark_name>/
        
        If evaluation.included_results is specified in config, loads only those specific runs.
        
        Returns:
            List of loaded result dictionaries
        """
        try:
            # Get dataset from config
            dataset = self.config.get('dataset', 'default_dataset')
            
            # Create path to benchmark results folder
            current_dir = Path.cwd()
            benchmark_dir = current_dir / dataset / self.name
            
            if not benchmark_dir.exists():
                logger.warning(f"Benchmark results directory does not exist: {benchmark_dir}")
                return []
            
            # Check if specific results are requested in evaluation config
            evaluation_config = self.config.get('evaluation', {})
            included_results = evaluation_config.get('included_results')
            
            # Handle string "None" or "null" as None
            if included_results in [None, "None", "null", ""]:
                included_results = None
            
            loaded_results = []
            
            if included_results:
                # Load specific results as configured
                logger.info(f"Loading specific results: {included_results}")
                for result_name in included_results:
                    result_dir = benchmark_dir / result_name
                    if result_dir.exists():
                        results_file = result_dir / "results.json"
                        if results_file.exists():
                            try:
                                with open(results_file, 'r') as f:
                                    result_data = json.load(f)
                                loaded_results.append(result_data)
                                logger.debug(f"Loaded result: {result_name}")
                            except Exception as e:
                                logger.error(f"Failed to load result {result_name}: {e}")
                        else:
                            logger.warning(f"Results file not found: {results_file}")
                    else:
                        logger.warning(f"Result directory not found: {result_dir}")
            else:
                # Load latest results for each flavour
                logger.info("Loading latest results for each flavour")
                
                # Group result directories by flavour
                flavour_latest = {}
                
                for item in benchmark_dir.iterdir():
                    if item.is_dir() and '_' in item.name:
                        # Skip plot directories and other non-result directories
                        if item.name.endswith('_plots') or item.name == 'data':
                            continue
                            
                        # Parse timestamp_flavour format: YYYYMMDD_HHMMSS_flavour
                        # Timestamp is always first 15 characters (YYYYMMDD_HHMMSS)
                        # Flavour is everything after that with underscore removed
                        if len(item.name) > 15 and item.name[15] == '_':
                            timestamp_part = item.name[:15]  # YYYYMMDD_HHMMSS
                            flavour = item.name[16:]  # Everything after timestamp_
                            
                            # Keep track of latest timestamp for each flavour
                            if flavour not in flavour_latest or timestamp_part > flavour_latest[flavour]['timestamp']:
                                flavour_latest[flavour] = {
                                    'timestamp': timestamp_part,
                                    'dir_name': item.name,
                                    'path': item
                                }
                
                # Load the latest result for each flavour
                for flavour, info in flavour_latest.items():
                    results_file = info['path'] / "results.json"
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                result_data = json.load(f)
                            loaded_results.append(result_data)
                            logger.debug(f"Loaded latest result for flavour '{flavour}': {info['dir_name']}")
                        except Exception as e:
                            logger.error(f"Failed to load result {info['dir_name']}: {e}")
                    else:
                        logger.warning(f"Results file not found: {results_file}")
            
            logger.info(f"Loaded {len(loaded_results)} benchmark results")
            return loaded_results
            
        except Exception as e:
            logger.error(f"Failed to load benchmark results: {e}")
            return []

    @abstractmethod
    def _evaluate_benchmark(self) -> None:
        """Evaluate benchmark results"""
        pass
