"""
Experiment runner utilities for MIRROR.

Provides load_experiment_config() which deep-merges a small experiment YAML
(containing only deltas) with the base config.yaml. This eliminates Kaggle
notebook config drift by making notebooks thin wrappers around experiment YAMLs.
"""

import yaml
from pathlib import Path


def load_experiment_config(
    experiment_path: str,
    base_config_path: str = "src/config.yaml",
) -> dict:
    """Load and merge an experiment config with the base config.

    Args:
        experiment_path: Path to experiment YAML (e.g., "src/experiment_configs/base_naked.yaml")
        base_config_path: Path to the base config.yaml (default: "src/config.yaml")

    Returns:
        Merged config dict with _experiment_name and _experiment_description set.
    """
    base_path = Path(base_config_path)
    exp_path = Path(experiment_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    with open(exp_path, "r", encoding="utf-8") as f:
        exp = yaml.safe_load(f)

    # Deep-merge overrides into base config
    for section, overrides in exp.get("overrides", {}).items():
        if section in base and isinstance(base[section], dict) and isinstance(overrides, dict):
            base[section].update(overrides)
        else:
            base[section] = overrides

    # Tag the config with experiment metadata
    base["_experiment_name"] = exp.get("name", "unknown")
    base["_experiment_description"] = exp.get("description", "")

    return base
