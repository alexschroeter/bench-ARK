"""
Main CLI entry point for bench-ARK.

This module provides the command-line interface for running benchmarks,
managing configurations, and analyzing results.
"""

import logging
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

import click


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise click.ClickException(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise click.ClickException(f"Error parsing configuration file: {e}")
    except Exception as e:
        raise click.ClickException(f"Error reading configuration file: {e}")


def set_nested_config_value(config: Dict[str, Any], key: str, value: Any) -> None:
    """Set a nested configuration value using dot notation."""
    keys = key.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        elif not isinstance(current[k], dict):
            # Convert non-dict values to dict if we need to nest deeper
            current[k] = {}
        current = current[k]
    
    # Set the final value
    current[keys[-1]] = value


def parse_override_value(value_str: str) -> Any:
    """Parse override value, attempting to convert to appropriate type."""
    # Try to parse as YAML for complex types
    try:
        return yaml.safe_load(value_str)
    except yaml.YAMLError:
        # If YAML parsing fails, return as string
        return value_str


@click.group()
@click.version_option(version="0.1.0", prog_name="bench-ark")
@click.option(
    "--verbose", "-v",
    count=True,
    help="Increase verbosity (use -v, -vv, or -vvv)"
)
@click.pass_context
def cli(ctx: click.Context, verbose: int) -> None:
    """
    bench-ARK: A comprehensive benchmarking framework for ARK models.
    
    Use this tool to run performance benchmarks, analyze results,
    and compare different configurations across various devices.
    """
    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)
    
    # Store verbose count for subcommands to use
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument(
    'config_file',
    type=click.Path(exists=True, path_type=Path),
    required=True
)
@click.option(
    '--set', '-s',
    multiple=True,
    help="Override config values using key=value syntax (e.g., -s benchmarks.enabled=[test_benchmark] -s output.results_dir=custom_results)"
)
@click.option(
    '--dry-run', '-n',
    is_flag=True,
    help="Show what would be done without executing"
)
@click.option(
    "--verbose", "-v",
    count=True,
    help="Increase verbosity (use -v, -vv, or -vvv) - can also be used as global option before 'run'"
)
@click.pass_context
def run(
    ctx: click.Context,
    config_file: Path,
    set: tuple,
    dry_run: bool,
    verbose: int
) -> None:
    """Run benchmarks using the specified configuration file."""
    
    # Load base configuration first to check for global logging settings
    config = load_config(config_file)
    
    # Apply overrides from --set options
    for override in set:
        if '=' not in override:
            raise click.ClickException(f"Invalid override format: {override}. Use key=value format.")
        
        key, value_str = override.split('=', 1)
        value = parse_override_value(value_str)
        set_nested_config_value(config, key, value)
    
    # Determine effective log level with precedence:
    # 1. Command-line verbose option (highest priority)
    # 2. Global verbose option 
    # 3. Config file logging.level
    # 4. Default WARNING (lowest priority)
    
    if verbose > 0:
        # Command-line verbose takes precedence
        effective_verbose = verbose
        if effective_verbose == 1:
            log_level = "INFO"
        else:
            log_level = "DEBUG"
        log_source = "command-line verbose"
    elif ctx.obj.get('verbose', 0) > 0:
        # Global verbose option
        global_verbose = ctx.obj.get('verbose', 0)
        if global_verbose == 1:
            log_level = "INFO"
        else:
            log_level = "DEBUG"
        log_source = "global verbose"
    else:
        # Check config file for logging level
        config_log_level = config.get('logging', {}).get('level')
        if config_log_level:
            log_level = config_log_level.upper()
            log_source = "config file"
        else:
            log_level = "WARNING"
            log_source = "default"
    
    # Setup global logging
    config_log_file = config.get('logging', {}).get('file')
    setup_logging(log_level, config_log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading configuration from: {config_file}")
    logger.info(f"Log level set to: {log_level} (source: {log_source})")
    if config_log_file:
        logger.info(f"Log file: {config_log_file}")
    
    if dry_run:
        logger.info("DRY RUN MODE - No benchmarks will be executed")
    
    # Log applied overrides
    for override in set:
        key, value_str = override.split('=', 1)
        value = parse_override_value(value_str)
        logger.info(f"Override applied: {key} = {value}")
    
    if dry_run:
        # Show what would be done
        click.echo("\n=== DRY RUN SUMMARY ===")
        click.echo(f"Configuration file: {config_file}")
        click.echo(f"Final configuration:")
        click.echo(yaml.dump(config, default_flow_style=False, indent=2))
        return
    
    try:
        # Initialize and run benchmark manager
        from ..core.benchmark_manager import BenchmarkManager
        
        logger.info("Starting benchmark execution...")
        benchmark_manager = BenchmarkManager(config)
                
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}", exc_info=True)
        raise click.ClickException(f"Benchmark execution failed: {e}")


@cli.command()
@click.argument(
    'config_file',
    type=click.Path(exists=True, path_type=Path),
    required=True
)
@click.option(
    '--set', '-s',
    multiple=True,
    help="Override config values using key=value syntax"
)
def show_config(
    config_file: Path,
    set: tuple,
) -> None:
    """Show the final configuration after applying overrides."""
    
    setup_logging("WARNING")  # Reduce noise for this command
    
    # Load base configuration
    config = load_config(config_file)
    
    # Apply overrides
    for override in set:
        if '=' not in override:
            raise click.ClickException(f"Invalid override format: {override}. Use key=value format.")
        
        key, value_str = override.split('=', 1)
        value = parse_override_value(value_str)
        set_nested_config_value(config, key, value)
    
    # Output configuration
    config_yaml = yaml.dump(config, default_flow_style=False, indent=2)

    click.echo("=== Final Configuration ===")
    click.echo(config_yaml)


def main():
    """Entry point for the bench-ark command-line tool."""
    cli()


if __name__ == "__main__":
    cli()
