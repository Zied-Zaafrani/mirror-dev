"""Create the Phase 7 Contraindication Prior Kaggle notebook."""
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
    "# MIRROR — Phase 7: Drug-Lab Contraindication Prior\n\n"
    "**Objective:** Ablate the application of a clinically-grounded knowledge prior that penalizes the logits "
    "of drugs known to be contraindicated with the patient's current abnormal lab states.\n\n"
    "**Clinical Rules Implemented:**\n"
    "- **Hyperkalemia (K > 5.0)**: Avoid ACEi, ARBs, K-sparing diuretics, NSAIDs, K-supplements.\n"
    "- **Hypokalemia (K < 3.5)**: Avoid Digoxin (toxicity risk).\n"
    "- **Renal Failure (Cr > 1.2)**: Avoid NSAIDs, Metformin, K-sparing diuretics.\n"
    "- **Hepatotoxicity (ALT/AST High)**: Avoid Paracetamol.\n"
    "- **Bleeding Risk (INR/PT/PTT High)**: Avoid Anticoagulants and NSAIDs.\n\n"
    "**Configs:**\n"
    "- `baseline_per_lab`: Standard `per_lab_attn` without prior\n"
    "- `contraind_prior_soft`: `per_lab_attn` WITH clinical prior (Soft Penalty: 5.0)\n"
    "- `contraind_prior_hard`: `per_lab_attn` WITH clinical prior (Hard Penalty: 20.0)\n"
))

cells.append(code_cell(r"""# Do NOT reinstall PyTorch — Kaggle ships CUDA-enabled PyTorch pre-installed.
import torch
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
!pip install -q torch_geometric
!pip install -q pyyaml pandas numpy scikit-learn transformers"""))

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

cells.append(code_cell(
    "# Define experiments\n"
    "EXPERIMENTS = [\n"
    "    ('--ablation no_ablation --lab_encoder_type per_lab_attn', 'baseline_per_lab'),\n"
    "    ('--ablation no_ablation --lab_encoder_type per_lab_attn --use_contraindication_prior --contraindication_penalty 5.0', 'contraind_prior_soft'),\n"
    "    ('--ablation no_ablation --lab_encoder_type per_lab_attn --use_contraindication_prior --contraindication_penalty 20.0', 'contraind_prior_hard'),\n"
    "]\n"
))

cells.append(code_cell(r"""# Run experiments
import subprocess, gc, torch
from pathlib import Path

reports_dir = Path("experiment_reports/active_runs/phase7_sweep/reports")
results_log = []

for args, name in EXPERIMENTS:
    print(f'\n{"="*60}')
    print(f'RUNNING: {name}')
    print(f'ARGS: {args}')
    print(f'{"="*60}\n')
    
    for seed in [42, 123, 456]:
        run_name = f"{name}_seed{seed}"
        run_output_dir = reports_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_output_dir / "training_log.txt"
        
        cmd = f'python -u src/train.py --config src/config.yaml {args} --encoder_type transformer --visit_level_training --seed {seed} --device cuda --results_dir {run_output_dir}'
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

reports_dir = Path("experiment_reports/active_runs/phase7_sweep/reports")

# Zip results for download (no .pt checkpoints)
zip_name = "reports_phase7_sweep.zip"
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

out = Path("notebooks/train_kaggle_phase7_contraind.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created {out} ({len(cells)} cells)")
