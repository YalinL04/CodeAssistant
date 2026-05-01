"""YAML configuration loader."""

import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_training_args(config: dict) -> dict:
    """Extract training arguments from config dict."""
    return config.get("training", {})


def get_model_args(config: dict) -> dict:
    """Extract model arguments from config dict."""
    return config.get("model", {})


def get_lora_args(config: dict) -> dict:
    """Extract LoRA arguments from config dict."""
    return config.get("lora", {})


def get_quantization_args(config: dict) -> dict:
    """Extract quantization arguments from config dict."""
    return config.get("quantization", {})
