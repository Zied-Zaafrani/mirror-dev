"""Assembles parts A+B+C into the final Kaggle notebook."""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
MIRROR = HERE.parent.parent.parent  # MIRROR root

# Run each part to generate cell JSON
for part in ["part_a","part_b","part_c"]:
    path = HERE / f"{part}.py"
    code = path.read_text(encoding="utf-8")
    ns = {}
    exec(compile(code, str(path), "exec"), ns)
    with open(HERE / f"cells_{part[-1]}.json","w") as f:
        json.dump(ns["CELLS"], f)
    print(f"  {part}: {len(ns['CELLS'])} cells")

# Assemble
all_cells = []
for part in ["a","b","c"]:
    fp = HERE / f"cells_{part}.json"
    if fp.exists():
        all_cells.extend(json.loads(fp.read_text()))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.11.0"}
    },
    "cells": all_cells
}

out = MIRROR / "notebooks" / "train_kaggle_supervisor_demo.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out,"w",encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\nNotebook: {out}")
print(f"Total cells: {len(all_cells)}")
print(f"Size: {out.stat().st_size//1024} KB")
