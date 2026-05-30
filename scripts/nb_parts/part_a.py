"""Part A: env setup + data exploration cells."""
GROQ_KEY = "YOUR_GROQ_API_KEY_HERE"

def cc(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[src]}
def md(src): return {"cell_type":"markdown","metadata":{},"source":[src]}

CELLS = []

# ── HEADER ────────────────────────────────────────────────────────────────────
CELLS.append(md("""# MIRROR — Supervisor Demo Notebook
## *Multimodal Intelligent Recommender for Rehabilitation and Outpatient Rx*

> **MIMIC-III | 6,350 patients | 131 ATC-3 drugs | Transformer + Visit-Level Training + FiLM Fusion**

This notebook documents the complete MIRROR thesis pipeline:
**Data → Modality Exploration → Training → Ablation → Metrics → Explainability → SOTA Comparison**

---
| Component | Detail |
|---|---|
| Dataset | MIMIC-III v1.4 (PhysioNet) |
| Drug vocabulary | 131 ATC-3 codes (Carmen/SafeDrug-compatible) |
| Split | hidr_vita_sequential (4:1:1 by patient) |
| Encoder | Transformer 2L + Visit-Level Training |
| Fusion | FiLM (Feature-wise Linear Modulation) |
| Best config | top_200_labs — Jaccard **0.5661** (seed 42) |
"""))

# ── SEC 0: ENVIRONMENT ─────────────────────────────────────────────────────────
CELLS.append(md("## Section 0 — Environment Setup"))
CELLS.append(cc(r"""import torch
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
!pip install -q torch_geometric groq
!pip install -q pyyaml pandas numpy scikit-learn matplotlib seaborn"""))

CELLS.append(cc(r"""import os, sys, glob, pickle, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if os.path.exists("/kaggle"):
    os.chdir("/kaggle/working")
    os.system("rm -rf ./data ./src")
    for d in ["data/processed","data/embeddings"]: os.makedirs(d, exist_ok=True)

    tp = glob.glob("/kaggle/input/**/train.py", recursive=True)
    if not tp: raise FileNotFoundError("train.py not found")
    os.system(f"cp -r {os.path.dirname(tp[0])} /kaggle/working/src")
    sys.path.append("/kaggle/working/src")

    for pat,dst in [("cohort_mimic3.pkl","data/processed"), ("lab_vectors_200labs.pkl","data/processed"),
                    ("lab_vectors_100labs.pkl","data/processed"), ("note_embeddings_mimic3.pkl","data/processed"),
                    ("notes_text_mimic3.pkl","data/processed"), ("ddi_A_final.pkl","data/processed"),
                    ("records_final.pkl","data/processed"), ("lab_data_mimic3.pkl","data/processed"),
                    ("voc_final.pkl","data/processed"), ("ehr_adj_final.pkl","data/processed"),
                    ("code_embeddings.pt","data/embeddings")]:
        hits = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
        if hits and not os.path.exists(f"./{dst}/{pat}"):
            os.symlink(hits[0], f"./{dst}/{pat}")

print("CWD:", os.getcwd())
print("Processed files:", sorted(os.listdir("data/processed")))"""))

# ── SEC 1: DATA EXPLORATION ───────────────────────────────────────────────────
CELLS.append(md("""## Section 1 — Data Modality Exploration
*What goes into MIRROR? This section visualises each input modality before any training.*"""))

CELLS.append(md("### 1A — Cohort Overview"))
CELLS.append(cc(r"""with open("data/processed/cohort_mimic3.pkl","rb") as f: cohort=pickle.load(f)
with open("data/processed/records_final.pkl","rb") as f: records=pickle.load(f)

pv = cohort["patient_visits"]
split = cohort["split_indices"]
drug_vocab = cohort["drug_vocab"]
idx2atc = {v:k for k,v in drug_vocab.items()}
total_adm = len(cohort["hadm_ids"])
total_pat = len(pv)
visits_per_pat = [len(v) for v in pv.values()]

print(f"Patients : {total_pat:,}")
print(f"Admissions : {total_adm:,}")
print(f"Train/Val/Test : {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")
print(f"Drug vocabulary : {cohort['num_drugs']} ATC-3 codes")
print(f"Diag vocabulary : {cohort['num_diag']} ICD-9 codes")
print(f"Visits/patient (median) : {int(np.median(visits_per_pat))}")
print(f"Visits/patient (max) : {max(visits_per_pat)}")

fig, axes = plt.subplots(1, 3, figsize=(16,4), facecolor="#0f1117")
plt.rcParams.update({"figure.facecolor":"#0f1117","axes.facecolor":"#1a1d2e",
    "axes.edgecolor":"#444","axes.labelcolor":"#e0e0e0",
    "xtick.color":"#aaa","ytick.color":"#aaa","text.color":"#e0e0e0","grid.color":"#2a2d3e"})

# split bar
ax=axes[0]; splits={"Train":len(split['train']),"Val":len(split['val']),"Test":len(split['test'])}
bars=ax.bar(list(splits.keys()),list(splits.values()),color=["#42a5f5","#ab47bc","#26a69a"],edgecolor="#0f1117",lw=1.5)
for bar,v in zip(bars,splits.values()): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+30,f"{v:,}",ha="center",fontsize=11,fontweight="bold")
ax.set_title("hidr_vita_sequential Split",fontsize=11); ax.set_ylim(0,max(splits.values())*1.15); ax.grid(axis="y",alpha=0.3)

# visit distribution
ax=axes[1]; ax.hist(visits_per_pat,bins=range(2,max(visits_per_pat)+2),color="#4fc3f7",edgecolor="#0f1117",alpha=0.85)
ax.axvline(np.median(visits_per_pat),color="#ffd54f",lw=2,ls="--",label=f"Median={int(np.median(visits_per_pat))}")
ax.set_xlabel("Visits per Patient"); ax.set_ylabel("Patients"); ax.set_title("Visit Distribution"); ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.3)

# drug frequency top 20
drug_counts=np.zeros(131)
for pat in records:
    for vis in pat:
        for d in vis[2]:
            if d<131: drug_counts[d]+=1
top20=np.argsort(drug_counts)[::-1][:20]
ax=axes[2]; colors_d=plt.cm.viridis(np.linspace(0.2,0.9,20))[::-1]
ax.barh([idx2atc.get(i,f"D{i}") for i in top20[::-1]],drug_counts[top20[::-1]],color=colors_d,edgecolor="#0f1117",lw=0.5)
ax.set_xlabel("Prescriptions (total)"); ax.set_title("Top 20 Drugs (ATC-3)"); ax.grid(axis="x",alpha=0.3)

plt.suptitle("MIRROR Cohort — MIMIC-III Statistics",fontsize=13,fontweight="bold",y=1.02)
plt.tight_layout(); plt.show()"""))

CELLS.append(md("### 1B — Laboratory Data (top_200_labs PKL)"))
CELLS.append(cc(r"""with open("data/processed/lab_vectors_200labs.pkl","rb") as f: lab200=pickle.load(f)
lab_names=lab200["lab_names"]; flags=lab200["lab_vectors"][:,200:]  # presence flags
coverage=flags.mean(axis=0)*100; N=flags.shape[0]

print(f"Labs included  : {len(lab_names)}")
print(f"Admissions     : {N:,}")
print(f"Mean coverage  : {coverage.mean():.1f}%")
print(f"Median coverage: {np.median(coverage):.1f}%")
print(f"Best lab       : {lab_names[coverage.argmax()]} ({coverage.max():.1f}%)")
print(f"Rarest lab     : {lab_names[coverage.argmin()]} ({coverage.min():.1f}%)")

top10=np.argsort(coverage)[::-1][:10]; bot10=np.argsort(coverage)[:10]
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(20,5),facecolor="#0f1117")

ct=plt.cm.Blues(np.linspace(0.4,0.9,10))[::-1]
ax1.barh([lab_names[i] for i in top10[::-1]],[coverage[i] for i in top10[::-1]],color=ct)
ax1.set_xlim(0,115)
for i_,idx in enumerate(top10[::-1]):
    ax1.text(coverage[idx]+0.5,i_,f"{coverage[idx]:.1f}%",va="center",fontsize=9)
ax1.set_xlabel("% Admissions Measured"); ax1.set_title("Top 10 Labs (Most Common)",color="#4fc3f7"); ax1.grid(axis="x",alpha=0.3)

cr=plt.cm.Reds(np.linspace(0.4,0.9,10))[::-1]
mx=max(coverage[i] for i in bot10)
ax2.barh([lab_names[i] for i in bot10[::-1]],[coverage[i] for i in bot10[::-1]],color=cr)
ax2.set_xlim(0,mx*1.35)
for i_,idx in enumerate(bot10[::-1]):
    ax2.text(coverage[idx]+mx*0.03,i_,f"{coverage[idx]:.1f}%",va="center",fontsize=9)
ax2.set_xlabel("% Admissions Measured"); ax2.set_title("Bottom 10 Labs (Rarest)",color="#ef9a9a"); ax2.grid(axis="x",alpha=0.3)

ax3.hist(coverage,bins=30,color="#26c6da",edgecolor="#0f1117",alpha=0.85)
ax3.axvline(coverage.mean(),color="#ffd54f",lw=2,ls="--",label=f"Mean {coverage.mean():.1f}%")
ax3.set_xlabel("Coverage (%)"); ax3.set_ylabel("# Lab Tests"); ax3.set_title("Coverage Distribution"); ax3.legend(); ax3.grid(axis="y",alpha=0.3)

plt.suptitle("Lab PKL — 200-Lab Configuration Coverage Analysis",fontsize=13,fontweight="bold",y=1.02)
plt.tight_layout(); plt.show()"""))

CELLS.append(md("### 1C — Clinical Notes"))
CELLS.append(cc(r"""with open("data/processed/note_embeddings_mimic3.pkl","rb") as f: notes=pickle.load(f)
has_note=notes["has_note"]; embs=notes["embeddings"]; method=notes["method"]
cov=has_note.mean()*100

print(f"Encoding method : {method}")
print(f"Embedding dim   : {embs.shape[1]}")
print(f"Total admissions: {len(has_note):,}")
print(f"With notes      : {int(has_note.sum()):,} ({cov:.1f}%)")
print(f"Without notes   : {int((1-has_note).sum()):,} ({100-cov:.1f}%)")

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5),facecolor="#0f1117")
ax1.pie([cov,100-cov],labels=[f"Has Note\n{cov:.1f}%",f"No Note\n{100-cov:.1f}%"],
    colors=["#42a5f5","#455a64"],autopct="%1.1f%%",startangle=90,
    textprops={"fontsize":11,"color":"#e0e0e0"},wedgeprops={"edgecolor":"#0f1117","linewidth":2})
ax1.set_title("ClinicalBERT Note Coverage\n(chunk+pool, 768-dim)",fontsize=12)

# PCA of embeddings to show structure
from sklearn.decomposition import PCA
sample_idx=np.random.choice(len(embs),min(2000,len(embs)),replace=False)
pca=PCA(n_components=2).fit_transform(embs[sample_idx])
sc=ax2.scatter(pca[:,0],pca[:,1],c=has_note[sample_idx],cmap="coolwarm",alpha=0.4,s=8)
plt.colorbar(sc,ax=ax2,label="Has Note (1=yes)")
ax2.set_xlabel("PCA-1"); ax2.set_ylabel("PCA-2")
ax2.set_title("Note Embedding Space (PCA — 2000 admissions)",fontsize=12)
ax2.grid(alpha=0.2)

plt.tight_layout(); plt.show()"""))

CELLS.append(md("### 1D — Drug Interaction Graph"))
CELLS.append(cc(r"""with open("data/processed/ddi_A_final.pkl","rb") as f: ddi=pickle.load(f)
ddi_mat=np.array(ddi)
print(f"Drug nodes : {ddi_mat.shape[0]}")
print(f"DDI pairs  : {int(ddi_mat.sum()//2):,}")
print(f"DDI density: {ddi_mat.mean()*100:.1f}%")

top30=np.argsort(drug_counts)[::-1][:30]
sub=ddi_mat[np.ix_(top30,top30)]; labs30=[idx2atc.get(i,f"D{i}") for i in top30]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,7),facecolor="#0f1117")
im=ax1.imshow(sub,cmap="RdYlGn_r",aspect="auto",vmin=0,vmax=1)
ax1.set_xticks(range(30)); ax1.set_xticklabels(labs30,rotation=45,ha="right",fontsize=8)
ax1.set_yticks(range(30)); ax1.set_yticklabels(labs30,fontsize=8)
plt.colorbar(im,ax=ax1,fraction=0.03,pad=0.02,label="DDI pair")
ax1.set_title("Drug-Drug Interaction Matrix (Top 30 drugs)",fontsize=12)

degrees=ddi_mat.sum(axis=1)
top_ddi=np.argsort(degrees)[::-1][:15]
ax2.barh([idx2atc.get(i,f"D{i}") for i in top_ddi[::-1]],degrees[top_ddi[::-1]],
    color=plt.cm.Reds(np.linspace(0.4,0.9,15))[::-1],edgecolor="#0f1117")
ax2.set_xlabel("Number of Known DDI Pairs"); ax2.set_title("Most Dangerous Drugs (DDI degree)",fontsize=12); ax2.grid(axis="x",alpha=0.3)

plt.tight_layout(); plt.show()"""))

CELLS.append(md("### 1E — Data Summary Card"))
_summary_cell = (
    'from IPython.display import HTML\n'
    'css = "<style>.mc{background:#1a1d2e;border-radius:12px;padding:20px;font-family:Arial;color:#e0e0e0;margin:10px 0}"\n'
    '  ".mc h3{color:#4fc3f7;margin-top:0} table.ct{width:100%;border-collapse:collapse}"\n'
    '  "table.ct th{background:#0f1117;color:#4fc3f7;padding:8px 12px;text-align:left}"\n'
    '  "table.ct td{padding:8px 12px;border-bottom:1px solid #2a2d3e}"\n'
    '  ".badge{background:#4fc3f7;color:#0f1117;border-radius:4px;padding:2px 7px;font-weight:bold}</style>"\n'
    'body = ("<div class=\\"mc\\"><h3>MIRROR \u2014 Multimodal Input Summary</h3><table class=\\"ct\\">"\n'
    '  "<tr><th>Modality</th><th>Type</th><th>Dimension</th><th>Coverage</th><th>Role</th></tr>"\n'
    '  "<tr><td><b>ICD-9 Codes</b></td><td>Multi-hot</td><td>1,958 diag + 1,430 proc</td>"\n'
    '  "  <td><span class=\\"badge\\">100%</span></td><td>Core backbone</td></tr>"\n'
    '  "<tr><td><b>Clinical Notes</b></td><td>ClinicalBERT 768d</td><td>768 per admission</td>"\n'
    '  "  <td>~87%</td><td>Precision signal (+0.025 Jac)</td></tr>"\n'
    '  "<tr><td><b>Lab Results</b></td><td>z-score + flag</td><td>400d (200\xd72)</td>"\n'
    '  "  <td>Varies (~95% for top labs)</td><td>Recall signal (+0.006 Jac marginal)</td></tr>"\n'
    '  "<tr><td><b>Drug Graph</b></td><td>HGT GNN</td><td>131 nodes \xd7 256d</td>"\n'
    '  "  <td><span class=\\"badge\\">100%</span></td><td>Drug structural priors</td></tr>"\n'
    '  "<tr><td><b>LLM Embeddings</b></td><td>PubMedBERT 768d</td><td>768 per drug</td>"\n'
    '  "  <td><span class=\\"badge\\">100%</span></td><td>Drug semantic init</td></tr>"\n'
    '  "</table></div>")\n'
    'display(HTML(css + body))\n'
)
CELLS.append(cc(_summary_cell))


if __name__ == "__main__":
    import json, pathlib
    pathlib.Path("nb_parts").mkdir(exist_ok=True)
    with open("nb_parts/cells_a.json","w") as f:
        json.dump(CELLS, f)
    print(f"Part A: {len(CELLS)} cells saved.")
