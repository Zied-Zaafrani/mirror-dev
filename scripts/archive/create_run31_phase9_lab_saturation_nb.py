"""Create the Phase 9 Lab Saturation Sweep Kaggle notebook."""
import json
from pathlib import Path

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source]
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    }

cells = []

cells.append(md_cell(
    "# MIRROR — Phase 9: Lab Density Saturation Sweep\n\n"
    "**Objective:** Find the true saturation point of lab information density.\n"
    "Phase 8 showed a strictly monotonic increase up to 100 labs. This sweep continues from 150 upward "
    "to the maximum possible lab count (204 labs) determined by local cohort audit.\n\n"
    "**Configs:**\n"
    "- `top_150`: 150 labs\n"
    "- `top_200`: 200 labs\n"
    "- `top_250`: 250 labs\n"
    "- `top_300`: 300 labs\n"
    "- `top_350`: 350 labs\n"
    "- `top_400`: 400 labs\n"
    "- `top_446_MAX`: 446 labs (Absolute Ceiling - 100% of available clinical labs)\n\n"
    "**SOTA Base:** Transformer + Visit-Level Training + FiLM Fusion (Run 41 Baseline)."
))

cells.append(code_cell(r"""# Do NOT reinstall PyTorch — Kaggle ships CUDA-enabled PyTorch pre-installed.
import torch
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
!pip install -q torch_geometric
!pip install -q pyyaml pandas numpy scikit-learn"""))

cells.append(code_cell(r"""import os, sys, glob

if os.path.exists("/kaggle"):
    print("Running on Kaggle")
    os.chdir("/kaggle/working")
    os.system("rm -rf ./data ./src")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)

    # Copy src code
    train_paths = glob.glob("/kaggle/input/**/train.py", recursive=True)
    if not train_paths:
        raise FileNotFoundError("train.py not found in /kaggle/input")
    src_dir = os.path.dirname(train_paths[0])
    print(f"Found src at: {src_dir}")
    os.system(f"cp -r {src_dir} /kaggle/working/src")
    sys.path.append("/kaggle/working/src")

    # Symlink processed data files
    processed_paths = glob.glob("/kaggle/input/**/cohort_mimic3.pkl", recursive=True)
    if not processed_paths:
        raise FileNotFoundError("cohort_mimic3.pkl not found in /kaggle/input")
    processed_dir = os.path.dirname(processed_paths[0])
    print(f"Found processed dir at: {processed_dir}")
    for fpath in glob.glob(f"{processed_dir}/*"):
        fname = os.path.basename(fpath)
        link = f"./data/processed/{fname}"
        if not os.path.exists(link):
            os.symlink(fpath, link)

    # Symlink embeddings
    emb_paths = glob.glob("/kaggle/input/**/code_embeddings.pt", recursive=True)
    if not emb_paths:
        raise FileNotFoundError("code_embeddings.pt not found in /kaggle/input")
    emb_dir = os.path.dirname(emb_paths[0])
    print(f"Found embeddings dir at: {emb_dir}")
    for fpath in glob.glob(f"{emb_dir}/*"):
        fname = os.path.basename(fpath)
        link = f"./data/embeddings/{fname}"
        if not os.path.exists(link):
            os.symlink(fpath, link)

print("Working directory:", os.getcwd())"""))

cells.append(code_cell(r"""# ── Verify all required Phase 9 PKL files are present ──
import pickle
from pathlib import Path

print("Scanning available lab PKL files...")
available_pkls = {}
for pkl_path in sorted(Path("data/processed").glob("lab_vectors_*labs.pkl")):
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        n = len(d["lab_names"])
        available_pkls[n] = str(pkl_path)
        print(f"  {pkl_path.name}: {n} labs")
    except Exception as e:
        print(f"  Error reading {pkl_path.name}: {e}")

# Build experiment list
# Target counts: 150, 200, 250, 300, 350, 400, 446
TARGETS = [150, 200, 250, 300, 350, 400, 446]
BASE_FLAGS = "--encoder_type transformer --visit_level_training --fusion_strategy film --no_diagnostics"

EXPERIMENTS = []
for n_labs in TARGETS:
    if n_labs in available_pkls:
        pkl = available_pkls[n_labs]
        args = f"{BASE_FLAGS} --num_labs {n_labs} --lab_pkl {pkl}"
        name = f"top_{n_labs}_labs"
        EXPERIMENTS.append((args, name))
    else:
        # Fallback: search for best match
        suitable = {k: v for k, v in available_pkls.items() if k >= n_labs}
        if suitable:
            best_k = min(suitable.keys())
            pkl = suitable[best_k]
            args = f"{BASE_FLAGS} --num_labs {n_labs} --lab_pkl {pkl}"
            name = f"top_{n_labs}_labs"
            EXPERIMENTS.append((args, name))
            print(f"  [Auto-matched] {n_labs} labs -> {pkl}")
        else:
            print(f"  [ERROR] No pkl covers {n_labs} labs!")

print(f"\nDefined {len(EXPERIMENTS)} Phase 9 experiments.")"""))

cells.append(code_cell(r"""# Run experiments
import subprocess, gc, torch
from pathlib import Path

reports_dir = Path("experiment_reports/active_runs/phase9_saturation/reports")
results_log = []
SEED = 42

for args, name in EXPERIMENTS:
    print(f'\n{"="*60}')
    print(f'RUNNING: {name}')
    print(f'ARGS: {args}')
    print(f'{"="*60}\n')
    
    run_name = f"{name}_seed{SEED}"
    run_output_dir = reports_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_output_dir / "training_log.txt"
    
    cmd = f'python -u src/train.py --config src/config.yaml {args} --seed {SEED} --device cuda --results_dir {run_output_dir}'
    print(f'>> {cmd}')
    
    try:
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                print(line, end="")
                lf.write(line)
            proc.wait()
        status = "SUCCESS" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        results_log.append(f"{status}: {run_name}")
    except Exception as e:
        results_log.append(f"CRASH: {run_name}: {e}")
        print(f"CRASH: {e}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n\n--- SWEEP COMPLETE ---")
for entry in results_log:
    print(entry)"""))

cells.append(code_cell(r"""import json, zipfile
import numpy as np
from pathlib import Path

reports_dir = Path("experiment_reports/active_runs/phase9_saturation/reports")

# Zip results for download (no .pt checkpoints)
zip_name = "reports_phase9_saturation.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(reports_dir.rglob("result_*.json")):
        zf.write(p, p.relative_to(reports_dir))
    for p in sorted(reports_dir.rglob("training_log.txt")):
        zf.write(p, p.relative_to(reports_dir))
n_json = sum(1 for _ in reports_dir.rglob("result_*.json"))
print(f"Zipped {n_json} result JSON(s) → {zip_name}")"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

out = Path("notebooks/train_kaggle_phase9_lab_saturation.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created {out} ({len(cells)} cells)")
