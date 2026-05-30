"""Part B: Training (Section 2), Graphs (Section 3), Ablation (Section 4)."""
import json, pathlib

def cc(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[src]}
def md(src): return {"cell_type":"markdown","metadata":{},"source":[src]}

CELLS = []

# ── SEC 2: 3-SEED TRAINING ────────────────────────────────────────────────────
CELLS.append(md("## Section 2 — top_200_labs: 3-Seed Replication\n*Seeds 42 (Phase 9), 123 and 456 (this run). Best seed used for ablation in Section 4.*"))

CELLS.append(cc(r"""import subprocess, gc, torch
from pathlib import Path

reports_dir = Path("experiment_reports/active_runs/phase10_200lab/reports")
SEEDS = [42, 123, 456]
RESULTS = {}

# Find lab PKL
import glob as glb
lab_pkls = sorted(glb.glob("data/processed/lab_vectors_200labs.pkl"))
if not lab_pkls: raise FileNotFoundError("lab_vectors_200labs.pkl not found")
lab_pkl = lab_pkls[0]
BASE_FLAGS = f"--encoder_type transformer --visit_level_training --fusion_strategy film --no_diagnostics --num_labs 200 --lab_pkl {lab_pkl} --soft_jaccard_weight 1.0 --bce_weight 0.5"

for seed in SEEDS:
    run_name = f"top_200_labs_seed{seed}"
    out_dir = reports_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.txt"
    print(f"\n{'='*55}\nRunning seed {seed}\n{'='*55}")
    cmd = f"python -u src/train.py --config src/config.yaml {BASE_FLAGS} --seed {seed} --device cuda --results_dir {out_dir} --save_checkpoint"
    with open(log_path,"w") as lf:
        proc = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout: print(line,end=""); lf.write(line)
        proc.wait()
    hits = list(out_dir.glob("result_*.json"))
    if hits:
        m = json.loads(hits[0].read_text())["test_metrics"]
        RESULTS[seed] = m
        print(f"  => Jaccard={m['Jaccard']:.4f}")
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n--- DONE ---")
for s,m in RESULTS.items(): print(f"  Seed {s}: Jac={m['Jaccard']:.4f} F1={m['F1']:.4f}")"""))

CELLS.append(cc(r"""import numpy as np, json
from pathlib import Path

# Include seed 42 from Phase 9 if not in RESULTS
SEED42_JAC = 0.5661  # Phase 9 single-seed result
reports_dir = Path("experiment_reports/active_runs/phase10_200lab/reports")
RESULTS_ALL = {}
for seed in [42,123,456]:
    hits = list((reports_dir/f"top_200_labs_seed{seed}").glob("result_*.json")) if (reports_dir/f"top_200_labs_seed{seed}").exists() else []
    if hits:
        RESULTS_ALL[seed] = json.loads(hits[0].read_text())["test_metrics"]

if 42 not in RESULTS_ALL:
    print(f"Seed 42 from Phase 9: {SEED42_JAC} (no checkpoint re-run)")

jacs = [RESULTS_ALL[s]["Jaccard"] for s in RESULTS_ALL] or [SEED42_JAC]
if 42 not in RESULTS_ALL: jacs = [SEED42_JAC] + jacs

BEST_SEED = list(RESULTS_ALL.keys())[np.argmax(jacs[1:] if 42 not in RESULTS_ALL else jacs)] if RESULTS_ALL else 42
print(f"\n3-seed Jaccard: {[f'{j:.4f}' for j in jacs]}")
print(f"Mean: {np.mean(jacs):.4f} ± {np.std(jacs,ddof=1):.4f}")
print(f"Best seed: {BEST_SEED}")
THRESHOLD = 0.5640
verdict = "NEW THESIS PEAK ★" if np.mean(jacs)>=THRESHOLD else "Use top_100_labs (0.5636±0.0025)"
print(f"Verdict: {verdict}")"""))

# ── SEC 3: GRAPHS ─────────────────────────────────────────────────────────────
CELLS.append(md("## Section 3 — Result Visualisations"))

CELLS.append(cc(r"""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.facecolor":"#0f1117","axes.facecolor":"#1a1d2e",
    "axes.edgecolor":"#444","axes.labelcolor":"#e0e0e0","xtick.color":"#aaa",
    "ytick.color":"#aaa","text.color":"#e0e0e0","grid.color":"#2a2d3e","grid.alpha":0.5})

# ── 3A: Lab Saturation Curve ──
LAB_SAT = {1:(0.5557,0.0017),5:(0.5569,0.001),10:(0.5568,0.0018),18:(0.5598,0.0003),
    20:(0.556,0.0007),30:(0.5594,0.0007),50:(0.5624,0.0002),100:(0.5636,0.0025),
    150:(0.5623,0.0),200:(0.5661,0.0),250:(0.5626,0.0),300:(0.5631,0.0),
    350:(0.5635,0.0),400:(0.564,0.0),446:(0.5645,0.0)}
# Update 200 with 3-seed result if available
if len(jacs)>=3: LAB_SAT[200]=(np.mean(jacs),np.std(jacs,ddof=1))
RUN41=0.5588
xs=sorted(LAB_SAT); ys=[LAB_SAT[x][0] for x in xs]; errs=[LAB_SAT[x][1] for x in xs]

fig,ax=plt.subplots(figsize=(13,5))
ax.plot(xs,ys,color="#4fc3f7",lw=2.5,marker="o",ms=6,zorder=3)
ax.fill_between(xs,[y-e for y,e in zip(ys,errs)],[y+e for y,e in zip(ys,errs)],alpha=0.2,color="#4fc3f7")
ax.axhline(RUN41,color="#ff7043",lw=1.5,ls="--"); ax.text(xs[-1]+3,RUN41+0.0003,"Run41 baseline\n0.5588",color="#ff7043",fontsize=9)
ax.scatter([200],[LAB_SAT[200][0]],color="#ffd54f",s=180,zorder=5,marker="*")
ax.annotate(f"Peak {LAB_SAT[200][0]:.4f}\n(200 labs)",xy=(200,LAB_SAT[200][0]),xytext=(230,0.565),fontsize=9,color="#ffd54f",arrowprops=dict(arrowstyle="->",color="#ffd54f"))
ax.set_xlabel("Number of Lab Tests",fontsize=12); ax.set_ylabel("Jaccard Index",fontsize=12)
ax.set_title("MIRROR Lab Density Saturation Curve\n(Transformer+VL | FiLM | MIMIC-III)",fontsize=13)
ax.set_xlim(-5,470); ax.set_ylim(0.551,0.572)
ax.set_xticks([1,5,10,18,50,100,150,200,250,300,350,400,446]); ax.tick_params(axis="x",rotation=45,labelsize=8)
ax.grid(axis="y",alpha=0.4); plt.tight_layout(); plt.show()"""))

CELLS.append(cc(r"""# ── 3B: Precision-Recall scatter ──
import numpy as np, matplotlib.pyplot as plt

configs = {
    "top_200\n(0.5661)":(0.693,0.762,0.5661,"#ffd54f"),
    "top_100\n(0.5636)":(0.681,0.772,0.5636,"#4fc3f7"),
    "top_50\n(0.5624)":(0.677,0.776,0.5624,"#80deea"),
    "Run41\n(0.5588)":(0.690,0.756,0.5588,"#ff7043"),
    "traj_lstm\n(0.5588)":(0.681,0.766,0.5588,"#ce93d8"),
    "contraind_hard\n(0.5536)":(0.673,0.766,0.5536,"#e53935"),
    "contraind_soft\n(0.5574)":(0.676,0.769,0.5574,"#ef9a9a"),
}
fig,ax=plt.subplots(figsize=(10,7))
for name,(prec,rec,jac,col) in configs.items():
    sz=300+(jac-0.550)*12000
    ax.scatter(prec,rec,s=sz,color=col,alpha=0.85,edgecolors="#0f1117",lw=1.5,zorder=3)
    ax.annotate(name,(prec,rec),xytext=(6,4),textcoords="offset points",fontsize=8.5,color=col)
ax.set_xlabel("Precision ↑",fontsize=12); ax.set_ylabel("Recall ↑",fontsize=12)
ax.set_title("Precision vs Recall (bubble=Jaccard)",fontsize=12)
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))

CELLS.append(cc(r"""# ── 3C: Per-metric bars for run41 seeds ──
import numpy as np, json, matplotlib.pyplot as plt
from pathlib import Path

result_files = sorted(Path("experiment_reports/active_runs/reports_run41_thesis_peak").rglob("result_*.json"))
metrics=[json.loads(p.read_text(encoding="utf-8"))["test_metrics"] for p in result_files]
seeds=[json.loads(p.read_text(encoding="utf-8"))["seed"] for p in result_files]
mk=["Jaccard","F1","PRAUC","Precision","Recall"]
x=np.arange(len(mk)); w=0.25
fig,ax=plt.subplots(figsize=(12,5))
for i,(seed,col) in enumerate(zip(seeds,["#4fc3f7","#ab47bc","#26a69a"])):
    vals=[metrics[i][k] for k in mk]
    ax.bar(x+i*w,vals,w,label=f"Seed {seed}",color=col,edgecolor="#0f1117",lw=0.8)
means=[np.mean([m[k] for m in metrics]) for k in mk]
ax.plot(x+w,means,"o--",color="#ffd54f",lw=2,ms=8,label="Mean",zorder=5)
for j,m in enumerate(means): ax.text(j+w,m+0.003,f"{m:.4f}",ha="center",fontsize=8,color="#ffd54f",fontweight="bold")
ax.set_xticks(x+w); ax.set_xticklabels(mk,fontsize=12)
ax.set_ylim(0.5,0.85); ax.set_ylabel("Score"); ax.set_title("Run41 Thesis Peak — 3-Seed Metrics")
ax.legend(fontsize=10); ax.grid(axis="y",alpha=0.3); plt.tight_layout(); plt.show()"""))

CELLS.append(cc(r"""# ── 3D: DDI vs Jaccard trade-off ──
import numpy as np, matplotlib.pyplot as plt

configs_ddi=[
    ("top_200_labs",0.5661,0.079,"#ffd54f"),
    ("Run41 baseline",0.5588,0.081,"#ff7043"),
    ("contraind_soft",0.5574,0.078,"#80deea"),
    ("traj_lstm",0.5588,0.079,"#ce93d8"),
    ("contraind_hard",0.5536,0.079,"#e53935"),
    ("lab_delta",0.5565,0.082,"#ef9a9a"),
]
fig,ax=plt.subplots(figsize=(10,6))
ax.axvspan(0,0.08,alpha=0.07,color="#4caf50",label="Safe zone (DDI<0.08)")
for name,jac,ddi,col in configs_ddi:
    ax.scatter(ddi,jac,s=180,color=col,edgecolors="#0f1117",lw=1.5,zorder=3)
    ax.annotate(name,(ddi,jac),xytext=(4,4),textcoords="offset points",fontsize=9,color=col)
ax.annotate("",xy=(0.076,0.567),xytext=(0.082,0.556),arrowprops=dict(arrowstyle="->",color="#aaa",lw=1.5))
ax.text(0.077,0.564,"ideal\ndirection",fontsize=8,color="#aaa",ha="center")
ax.set_xlabel("DDI Rate ↓ (lower=safer)",fontsize=12); ax.set_ylabel("Jaccard ↑",fontsize=12)
ax.set_title("Safety vs Accuracy Trade-off",fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))

# ── SEC 4: MODALITY ABLATION ──────────────────────────────────────────────────
CELLS.append(md("""## Section 4 — Modality Contribution Ablation
*2×2 factorial: (Full System / Naked) × (Notes only / Labs only / Notes+Labs)*

**Full system**: Transformer+VL + FiLM + Copy + HGT GNN  
**Naked system**: same encoder, no Copy, no GNN (hgt_layers=0)  
Single seed = the best seed from Section 2.
"""))

CELLS.append(cc(r"""import yaml, copy, json, subprocess, gc, torch
from pathlib import Path

# Best seed from training
best_seed = BEST_SEED  # set in Section 2

# Load base config
with open("src/config.yaml") as f: base_cfg = yaml.safe_load(f)

def write_cfg(overrides, name):
    cfg = copy.deepcopy(base_cfg)
    for k,v in overrides.items():
        keys = k.split(".")
        d = cfg
        for kk in keys[:-1]: d = d[kk]
        d[keys[-1]] = v
    p = Path(f"ablation_configs/{name}.yaml")
    p.parent.mkdir(exist_ok=True)
    with open(p,"w") as f: yaml.dump(cfg, f)
    return str(p)

ABLATIONS = {
    # Full system configs
    "full_notes_labs":  {},  # baseline, all ON
    "full_notes_only":  {"preprocessing.lab_dim": 0},
    "full_labs_only":   {"preprocessing.note_method": "none"},
    # Naked system configs (no copy, no GNN)
    "naked_notes_labs": {"model.copy_mechanism": False, "model.hgt_layers": 0},
    "naked_notes_only": {"model.copy_mechanism": False, "model.hgt_layers": 0, "preprocessing.lab_dim": 0},
    "naked_labs_only":  {"model.copy_mechanism": False, "model.hgt_layers": 0, "preprocessing.note_method": "none"},
}

results_dir = Path("experiment_reports/active_runs/modality_ablation")
ABL_RESULTS = {}

for name, overrides in ABLATIONS.items():
    cfg_path = write_cfg(overrides, name)
    out_dir = results_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    cmd = (f"python -u src/train.py --config {cfg_path} "
           f"--encoder_type transformer --visit_level_training "
           f"--fusion_strategy film --no_diagnostics "
           f"--seed {best_seed} --device cuda --results_dir {out_dir}")
    log = out_dir/"training_log.txt"
    with open(log,"w") as lf:
        proc=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout: print(line,end=""); lf.write(line)
        proc.wait()
    hits=list(out_dir.glob("result_*.json"))
    if hits:
        m=json.loads(hits[0].read_text())["test_metrics"]
        ABL_RESULTS[name]=m
        print(f"  => Jac={m['Jaccard']:.4f} Prec={m['Precision']:.4f} Rec={m['Recall']:.4f}")
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n--- ABLATION COMPLETE ---")"""))

CELLS.append(cc(r"""import numpy as np, matplotlib.pyplot as plt
from IPython.display import HTML

rows = []
ORDER = ["full_notes_labs","full_notes_only","full_labs_only","naked_notes_labs","naked_notes_only","naked_labs_only"]
LABELS = {"full_notes_labs":"Full: Notes+Labs (BEST)","full_notes_only":"Full: Notes only",
    "full_labs_only":"Full: Labs only","naked_notes_labs":"Naked: Notes+Labs",
    "naked_notes_only":"Naked: Notes only","naked_labs_only":"Naked: Labs only"}

for name in ORDER:
    if name not in ABL_RESULTS: continue
    m=ABL_RESULTS[name]; g="full" if "full" in name else "naked"
    col="#4fc3f7" if g=="full" else "#ce93d8"
    rows.append(f'<tr><td><b style="color:{col}">{LABELS[name]}</b></td>'
                f'<td>{m["Jaccard"]:.4f}</td><td>{m["F1"]:.4f}</td><td>{m["PRAUC"]:.4f}</td>'
                f'<td>{m["DDI Rate"]:.4f}</td><td>{m["Precision"]:.4f}</td><td>{m["Recall"]:.4f}</td></tr>')

css = ('<style>.at{background:#1a1d2e;border-radius:8px;padding:16px;font-family:Arial;color:#e0e0e0}'
       'table.at th{background:#0f1117;color:#4fc3f7;padding:8px 12px}'
       'table.at td{padding:8px 12px;border-bottom:1px solid #2a2d3e}</style>')
header = '<div class="at"><h3 style="color:#ffd54f">Modality Ablation Results - Seed ' + str(best_seed) + '</h3>'
table = '<table class="at"><tr><th>Config</th><th>Jaccard</th><th>F1</th><th>PRAUC</th><th>DDI</th><th>Precision</th><th>Recall</th></tr>'
html = css + header + table + "".join(rows) + "</table></div>"
display(HTML(html))

if ABL_RESULTS:
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    groups={"Full System":["full_notes_labs","full_notes_only","full_labs_only"],
            "Naked System":["naked_notes_labs","naked_notes_only","naked_labs_only"]}
    gl=["Notes+Labs","Notes only","Labs only"]; gc=["#42a5f5","#66bb6a","#ffa726"]
    for ax,(gname,names) in zip(axes,groups.items()):
        jacs=[ABL_RESULTS.get(n,{}).get("Jaccard",0) for n in names]
        bars=ax.bar(gl,jacs,color=gc,edgecolor="#0f1117",lw=1.5,width=0.5)
        for bar,val in zip(bars,jacs):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,f"{val:.4f}",ha="center",fontsize=11,fontweight="bold",color="#ffd54f")
        lo=min(jacs)*0.97 if jacs else 0.5; hi=max(jacs)*1.02 if jacs else 0.6
        ax.set_ylim(lo,hi); ax.set_ylabel("Jaccard"); ax.set_title(gname,fontsize=13); ax.grid(axis="y",alpha=0.3)
    plt.suptitle("Modality Contribution Ablation (Seed "+str(best_seed)+")",fontsize=14,fontweight="bold",y=1.02)
    plt.tight_layout(); plt.show()

    if all(n in ABL_RESULTS for n in ["full_notes_labs","full_notes_only","naked_notes_labs","naked_labs_only"]):
        full_lab_lift = ABL_RESULTS["full_notes_labs"]["Jaccard"] - ABL_RESULTS["full_notes_only"]["Jaccard"]
        naked_lab_lift = ABL_RESULTS["naked_notes_labs"]["Jaccard"] - ABL_RESULTS["naked_labs_only"]["Jaccard"]
        print(f"Labs lift in full system : +{full_lab_lift:.4f}")
        print(f"Labs lift in naked system: +{naked_lab_lift:.4f}")
        if full_lab_lift >= naked_lab_lift*0.8:
            print("THESIS SUPPORTED: Full system does not diminish modality contributions")
        else:
            print("WARNING: Modalities partially overlap with system components - review thesis framing")"""))


for name in ORDER:
    if name not in ABL_RESULTS: continue
    m=ABL_RESULTS[name]; g="full" if "full" in name else "naked"
    rows.append(f'<tr><td><b style="color:{"#4fc3f7" if g=="full" else "#ce93d8"}">{LABELS[name]}</b></td>'
                f'<td>{m["Jaccard"]:.4f}</td><td>{m["F1"]:.4f}</td><td>{m["PRAUC"]:.4f}</td>'
                f'<td>{m["DDI Rate"]:.4f}</td><td>{m["Precision"]:.4f}</td><td>{m["Recall"]:.4f}</td></tr>')

html=f"""<style>.at{{background:#1a1d2e;border-radius:8px;padding:16px;font-family:Arial;color:#e0e0e0}}
table.at th{{background:#0f1117;color:#4fc3f7;padding:8px 12px}}table.at td{{padding:8px 12px;border-bottom:1px solid #2a2d3e}}</style>
<div class="at"><h3 style="color:#ffd54f">Modality Ablation — Seed {best_seed}</h3>
<table class="at"><tr><th>Config</th><th>Jaccard</th><th>F1</th><th>PRAUC</th><th>DDI</th><th>Precision</th><th>Recall</th></tr>
{''.join(rows)}</table></div>"""
display(HTML(html))

# Grouped bar chart
if ABL_RESULTS:
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    groups={"Full System":["full_notes_labs","full_notes_only","full_labs_only"],
            "Naked System":["naked_notes_labs","naked_notes_only","naked_labs_only"]}
    gl=["Notes+Labs","Notes only","Labs only"]; gc=["#42a5f5","#66bb6a","#ffa726"]
    for ax,(gname,names) in zip(axes,groups.items()):
        jacs=[ABL_RESULTS.get(n,{}).get("Jaccard",0) for n in names]
        bars=ax.bar(gl,jacs,color=gc,edgecolor="#0f1117",lw=1.5,width=0.5)
        for bar,val in zip(bars,jacs):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,f"{val:.4f}",ha="center",fontsize=11,fontweight="bold",color="#ffd54f")
        ax.set_ylim(min(jacs)*0.97 if jacs else 0.5,max(jacs)*1.02 if jacs else 0.6)
        ax.set_ylabel("Jaccard"); ax.set_title(gname,fontsize=13); ax.grid(axis="y",alpha=0.3)
    plt.suptitle(f"Modality Contribution Ablation (Seed {best_seed})",fontsize=14,fontweight="bold",y=1.02)
    plt.tight_layout(); plt.show()

    # Thesis test
    if all(n in ABL_RESULTS for n in ["full_notes_labs","full_notes_only","naked_notes_labs","naked_labs_only"]):
        full_lab_lift = ABL_RESULTS["full_notes_labs"]["Jaccard"] - ABL_RESULTS["full_notes_only"]["Jaccard"]
        naked_lab_lift = ABL_RESULTS["naked_notes_labs"]["Jaccard"] - ABL_RESULTS["naked_labs_only"]["Jaccard"]
        print(f"\nLabs lift in full system : +{full_lab_lift:.4f}")
        print(f"Labs lift in naked system: +{naked_lab_lift:.4f}")
        if full_lab_lift >= naked_lab_lift*0.8:
            print("✅ THESIS SUPPORTED: Full system does not diminish modality contributions")
        else:
            print("⚠️  Modalities partially overlap with system components — review thesis framing")"""))

if __name__ == "__main__":
    pathlib.Path("nb_parts").mkdir(exist_ok=True)
    with open("nb_parts/cells_b.json","w") as f:
        json.dump(CELLS, f)
    print(f"Part B: {len(CELLS)} cells saved.")
