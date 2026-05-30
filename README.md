# MIRROR — Experimental Development Codebase

This repository contains the full development and experimentation codebase for the MIRROR drug recommendation system, including ablation variants, pretraining utilities, and multi-cohort evaluation infrastructure.

> **For the clean champion-only codebase**, see the `src_final` repository.

---

## Contents

```
src/
├── train.py                    # Main training script (multi-cohort, full ablation CLI)
├── evaluate.py                 # Evaluation metrics
├── dataset.py                  # MIRRORDataset
├── split_protocol.py           # Train/val/test split (HI-DR/VITA-compatible)
├── config.yaml                 # Default hyperparameters
│
├── pretrain.py                 # Pretraining utilities
├── compute_similarity.py       # Patient similarity computation
├── extract_embeddings_pretrain.py
│
├── experiment_configs/         # Per-cohort YAML configs for ablation sweeps
│   └── *.yaml
│
├── model/
│   ├── model.py                # MIRROR top-level module
│   ├── visit_encoder.py
│   ├── historical_attention.py
│   ├── predictor.py
│   ├── registry.py
│   │
│   ├── graph_encoders/
│   ├── temporal_encoders/
│   ├── fusion_modules/
│   ├── decoders/
│   ├── lab_encoders/
│   ├── aggregators/
│   └── losses/
│
└── preprocess/
    ├── preprocess_mimic3.py
    ├── preprocess_mimic4.py
    ├── extract_notes.py
    ├── extract_labs.py
    ├── generate_code_embeddings.py
    └── lab_ranges.py
```

---

## Training

```bash
# Champion run (MIMIC-III)
python train.py --seed 42 --ddi_alpha 0.2

# Using a config YAML (ablation notebook style)
python train.py --config experiment_configs/mimic3_full.yaml --seed 42 --ddi_alpha 0.2

# MIMIC-IV variants
python train.py --mimic_version 4 --seed 42 --ddi_alpha 0.2
python train.py --mimic_version 4 --mimic4_sota --seed 42 --ddi_alpha 0.2
python train.py --mimic_version 4 --mimic4_full --seed 42 --ddi_alpha 0.2
```

---

## Requirements

```bash
conda create -n mirror python=3.10
conda activate mirror
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric transformers scikit-learn rdkit numpy pandas tqdm
```
