# Experiment Config System

Each experiment is defined by a small YAML file that specifies ONLY its delta from the base `config.yaml`.

## Format

```yaml
name: "r31_gat_normalized"
description: "GAT with frequency-normalized co-occur weights"
base: "config.yaml"  # always
overrides:
  model:
    gnn_type: "gat"
    use_notes: false
    use_labs: false
    use_copy: false
```

## How It Works

1. The experiment runner loads the base `config.yaml` (386 lines of defaults).
2. It loads the experiment YAML and deep-merges `overrides` into the base config.
3. The merged config is passed to the training loop.

This eliminates Kaggle notebook config drift — notebooks become thin wrappers
that just load an experiment YAML and call `train()`.

## Usage

```python
from experiment_runner import load_experiment_config

config = load_experiment_config("src/experiment_configs/base_naked.yaml")
# config is now the full merged dictionary with _experiment_name set
```

## Files

- `base_naked.yaml` — Phase 4 validated naked baseline (codes only)
- `base_notes_film.yaml` — Phase 4 validated codes + notes with FiLM fusion
- `base_notes_labs_film.yaml` — Phase 4 validated full multimodal
