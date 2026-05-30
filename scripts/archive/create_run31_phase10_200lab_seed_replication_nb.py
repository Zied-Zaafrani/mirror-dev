"""Create the Phase 10: top_200_labs Seed Replication Kaggle notebook.

This runs the top_200_labs config with seeds 123 and 456 to validate the
0.5661 single-seed (42) result found in the Phase 9 saturation sweep.
If the 3-seed mean holds >= 0.5640, this becomes the new thesis peak.
"""
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
    "# MIRROR — Phase 10: top_200_labs Seed Replication\n\n"
    "**Objective:** Validate the 0.5661 single-seed (42) Jaccard from top_200_labs.\n\n"
    "Phase 9 found that top_200_labs achieves the highest single-run Jaccard in MIRROR history.\n"
    "This notebook runs seeds 123 and 456 to produce a 3-seed mean ± std.\n\n"
    "**Target:** 3-seed mean >= 0.5640 confirms new thesis peak.\n\n"
    "**Config:** Transformer + Visit-Level Training + FiLM Fusion + 200 labs (Run 41 backbone)."
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

cells.append(code_cell(r"""# ── Verify top_200_labs PKL exists ──
import pickle
from pathlib import Path

lab_pkl = None
for pkl_path in sorted(Path("data/processed").glob("lab_vectors_*labs.pkl")):
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        n = len(d["lab_names"])
        print(f"  {pkl_path.name}: {n} labs")
        if n >= 200:
            lab_pkl = str(pkl_path)
    except Exception as e:
        print(f"  Error reading {pkl_path.name}: {e}")

if lab_pkl is None:
    raise FileNotFoundError("No lab PKL with >= 200 labs found!")
print(f"\nUsing: {lab_pkl}")

BASE_FLAGS = f"--encoder_type transformer --visit_level_training --fusion_strategy film --no_diagnostics --num_labs 200 --lab_pkl {lab_pkl}"
print(f"Base flags: {BASE_FLAGS}")"""))

cells.append(code_cell(r"""# Run seed replication (seeds 123 and 456 — seed 42 already done)
import subprocess, gc, torch
from pathlib import Path

reports_dir = Path("experiment_reports/active_runs/phase10_200lab_replication/reports")
results_log = []
SEEDS = [123, 456]

for seed in SEEDS:
    run_name = f"top_200_labs_seed{seed}"
    run_output_dir = reports_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_output_dir / "training_log.txt"
    
    print(f'\n{"="*60}')
    print(f'RUNNING: {run_name}')
    print(f'{"="*60}\n')
    
    cmd = f'python -u src/train.py --config src/config.yaml {BASE_FLAGS} --seed {seed} --device cuda --results_dir {run_output_dir}'
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

print("\n\n--- SEED REPLICATION COMPLETE ---")
for entry in results_log:
    print(entry)"""))

cells.append(code_cell(r"""# ── Compile 3-seed results (include seed 42 from Phase 9) ──
import json, numpy as np
from pathlib import Path

reports_dir = Path("experiment_reports/active_runs/phase10_200lab_replication/reports")

# Gather all result JSONs
results = {}
for json_path in sorted(reports_dir.rglob("result_*.json")):
    with open(json_path) as f:
        d = json.load(f)
    seed = d.get("seed", json_path.stem.split("_")[1].replace("seed",""))
    results[seed] = d
    print(f"  Seed {seed}: Jaccard={d.get('jaccard', d.get('Jaccard', 'N/A'))}")

# Add seed 42 from Phase 9 if not present
SEED42_JAC = 0.5661  # From Phase 9 sweep
if 42 not in results and "42" not in results:
    print(f"\n  [INFO] Seed 42 from Phase 9: Jaccard={SEED42_JAC}")

# Compute 3-seed stats
jacs = [SEED42_JAC]  # seed 42 from Phase 9
for seed, d in sorted(results.items()):
    jac = d.get("jaccard", d.get("Jaccard"))
    if jac:
        jacs.append(jac)

print(f"\n{'='*50}")
print(f"top_200_labs: {len(jacs)}-seed results")
print(f"  Jaccard values: {[f'{j:.4f}' for j in jacs]}")
print(f"  Mean:  {np.mean(jacs):.4f}")
print(f"  Std:   {np.std(jacs, ddof=1):.4f}")
print(f"  Range: {min(jacs):.4f} – {max(jacs):.4f}")
print(f"{'='*50}")

threshold = 0.5640
if np.mean(jacs) >= threshold:
    print(f"\n  ★ NEW THESIS PEAK CONFIRMED: {np.mean(jacs):.4f} >= {threshold}")
else:
    print(f"\n  ⚠ Below threshold: {np.mean(jacs):.4f} < {threshold}")
    print(f"  Fallback: use top_100_labs (0.5636 ± 0.0025)")"""))

cells.append(code_cell(r"""import json, zipfile
from pathlib import Path

reports_dir = Path("experiment_reports/active_runs/phase10_200lab_replication/reports")

# Zip results for download
zip_name = "reports_phase10_200lab_replication.zip"
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

out = Path("notebooks/train_kaggle_phase10_200lab_replication.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created {out} ({len(cells)} cells)")
