"""Utility functions for the LeafLens-AI project."""
import yaml
import os
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config/config.yaml
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_value(config: dict, key_path: str, default=None):
    """Get nested value from config dict using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'data.batch_size')
        default: Default value if key not found
        
    Returns:
        Value from config or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


__all__ = ["load_config", "get_config_value"]
