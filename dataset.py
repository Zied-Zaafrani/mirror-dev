"""
Dataset and DataLoader for MIRROR drug recommendation.

Knowledge-Grounded Drug Recommendation via GNNs and LLMs (MIRROR).

Patient record format:
  Each record is a list of visits: [[diag_list, proc_list, med_list, hadm_id], ...]
  This format is shared across HI-DR (data_loader_new.py), VITA (data_loader_new.py),
  COGNet (data_loader.py), SafeDrug, and GAMENet — all use the same MIMIC-III
  preprocessing convention. MIRROR extends this with:
    - note_embed: per-visit ClinicalBERT chunk-pooled embedding (768d)
    - lab_vector: per-visit standardized lab panel (2×N_labs: z-scores + missingness)
    - drug_history: OR-aggregated prior prescription history (num_drugs,)
    - med_per_visit: per-visit binary medication vectors (T, num_drugs)

The "one sample per non-first visit" expansion (eval_mode=True) mirrors
HI-DR/VITA/COGNet evaluation protocol where each patient contributes
(T-1) independent prediction tasks.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Standard clinical reference ranges (low, high)
LAB_CLINICAL_THRESHOLDS = {
    "Creatinine": (0.6, 1.2),
    "BUN": (7, 20),
    "ALT": (7, 56),
    "AST": (8, 48),
    "Bilirubin": (0.1, 1.2),
    "Alk Phos": (44, 147),
    "INR": (0.8, 1.2),
    "PT": (11.0, 13.5),
    "PTT": (25, 35),
    "Sodium": (135, 145),
    "Potassium": (3.5, 5.0),
    "Magnesium": (1.7, 2.2),
    "Calcium": (8.5, 10.5),
    "Glucose": (70, 100),
    "Albumin": (3.4, 5.4),
    "Lactate": (0.5, 2.2),
    "WBC": (4.5, 11.0),
    "Hemoglobin": (12.0, 17.5)
}

def compute_lab_bins(lab_vector: np.ndarray, means: np.ndarray, stds: np.ndarray, lab_names: list[str]) -> np.ndarray:
    """De-standardize Z-scores and map them to discrete bins: 0 (missing), 1 (low), 2 (normal), 3 (high).
    
    Args:
        lab_vector: (lab_dim,) where [0:N] are z-scores, [N:2N] are missingness flags (0=present, 1=missing)
        means: (N,) array of training set z-score means
        stds: (N,) array of training set z-score stds
        lab_names: (N,) list of lab names matching LAB_CLINICAL_THRESHOLDS keys
        
    Returns:
        bins: (N,) array of integer bins.
    """
    total_dim = len(lab_vector)
    # We expect even dimension for value+flag or value+flag+slope+var
    # For now, we assume [0:N] values, [N:2N] flags.
    n_labs = len(lab_names)
    bins = np.zeros(n_labs, dtype=np.int64)
    
    if total_dim < n_labs * 2:
        return bins

    z_scores = lab_vector[:n_labs]
    flags = lab_vector[n_labs : 2*n_labs]

    # FIX-B43: detect the fully-missing fallback case where dataset emits
    # lab_vec = zeros (flags=0 for all labs because the patient was not in the
    # lookup). Without this, every lab gets bin=2 ("normal") and the
    # ContraindicationPrior thinks the patient has perfect labs.
    if (flags == 0).all() and (z_scores == 0).all():
        # Treat all labs as MISSING (bin 0), not present-normal.
        return bins  # bins is already all zeros

    # FIX-CLIP-Z: clip extreme outliers (max observed +286 σ swamps Linear).
    z_scores = np.clip(z_scores, -5.0, 5.0)

    # Destandardize: raw = z * std + mean
    raw_values = (z_scores * stds) + means

    for i, name in enumerate(lab_names):
        # Missing (flag == 1) -> Bin 0
        if flags[i] >= 0.5:
            bins[i] = 0
            continue
            
        # Clinical thresholds only exist for the core 18 labs
        if name in LAB_CLINICAL_THRESHOLDS:
            low, high = LAB_CLINICAL_THRESHOLDS[name]
            val = raw_values[i]
            
            if val < low:
                bins[i] = 1 # Low
            elif val > high:
                bins[i] = 3 # High
            else:
                bins[i] = 2 # Normal
        else:
            # For non-core labs, just distinguish between present (2) and missing (0)
            bins[i] = 2
            
    return bins



class MIRRORDataset(Dataset):
    """Dataset for MIRROR drug recommendation.

    Each sample = one patient prefix, with one non-first visit as prediction target.

    Loads:
            - records: patient visit sequences [diag_idx, proc_idx, med_idx, hadm_id]
      - note embeddings: per-admission 768d vectors
      - lab vectors: per-admission 400d vectors (200 labs)
      - drug history: binary OR of all previous visits' med labels
    """

    def __init__(
        self,
        records: list,           # list of patients; each patient = list of visits
        cohort: dict,            # cohort metadata with hadm_ids
        note_data: dict | None,  # {embeddings, has_note, hadm_ids}
        lab_data: dict | None,   # {lab_vectors, hadm_ids}
        num_drugs: int = 131,
        lab_key: str = "lab_vectors",  # "lab_vectors" (36d) or "lab_vectors_72d" (72d)
        # Run 17 Tier 1: recency-weighted drug history instead of binary OR.
        # drug_history[d] = sum(0.9^(T-1-t) * med_labels[t][d]) — recent = weight 1.0.
        # Gives the model a signal about how recently each drug was prescribed.
        use_temporal_decay: bool = False,
        decay_rate: float = 0.9,
        # Run 17 Tier 1: mean-pool note embeddings from ALL history visits.
        # Passes alongside current note to give the model longitudinal note context.
        use_history_notes: bool = False,
        # Phase 4: drug training frequency for rare-first AR sequence ordering.
        # drug_freq[d] = number of times drug d appears in training visits.
        # If provided, __getitem__ returns med_sequence sorted ascending (rarest first).
        drug_freq: np.ndarray | None = None,
        # Phase 5.5: Visit-level unrolling (DrugDoctor Config 12)
        visit_level_scramble: bool = False,
        # Phase 5.4 retrieval: cross-patient health-status-aware retrieval data.
        # Can be a single dict {hadm_id: {reprs, multihots}} or a list of dicts for multi-channel.
        retrieval_data: dict | list[dict] | None = None,
        # Phase 7: Dynamic lab count sweep (Configs 21-25)
        num_labs: int = 200,
        # Phase 7: Lab Trajectory
        use_lab_trajectory: bool = False,
        # FIX user-#5 ablation: when True, zero out the z-score block so the
        # lab encoder only sees the missingness/presence flags. Used by the
        # `labs_presence_only` ablation to test whether lab VALUES (not just
        # ORDERING) carry signal.
        lab_values_zeroed: bool = False,
    ):
        self.records = records
        self.num_drugs = num_drugs
        self.use_temporal_decay = use_temporal_decay
        self.decay_rate = decay_rate
        self.use_history_notes = use_history_notes
        self.drug_freq = drug_freq          # (num_drugs,) float or None
        self.visit_level_scramble = visit_level_scramble
        self.num_labs = num_labs
        self.use_lab_trajectory = use_lab_trajectory
        self.lab_values_zeroed = lab_values_zeroed
        self.retrieval_data_list = []
        self.retrieval_k = 0
        self.retrieval_hidden_dim = 256
        self.retrieval_meta_list = []

        if retrieval_data:
            if isinstance(retrieval_data, dict):
                retrieval_data_list = [retrieval_data]
            else:
                retrieval_data_list = retrieval_data

            for rd_idx, rd in enumerate(retrieval_data_list):
                meta = rd.get("__meta__")
                if isinstance(meta, dict):
                    self.retrieval_meta_list.append(dict(meta))
                    logger.info("Retrieval index [%d] meta: top_k=%s, split=%s", rd_idx, meta.get("top_k"), meta.get("split_source"))
                
                valid_entries = {}
                expected_k, expected_h = None, None
                for raw_hadm, raw_entry in rd.items():
                    if isinstance(raw_hadm, str) and raw_hadm.startswith("__"): continue
                    try:
                        hadm_id = int(raw_hadm)
                        if not isinstance(raw_entry, dict): continue
                        sr = np.asarray(raw_entry["similar_reprs"], dtype=np.float32)
                        smh = np.asarray(raw_entry["similar_multihots"], dtype=np.float32)
                        if sr.ndim != 2 or smh.ndim != 2: continue
                        k, h = sr.shape
                        if expected_k is None: expected_k, expected_h = k, h
                        elif k != expected_k or h != expected_h: continue
                        valid_entries[hadm_id] = {"similar_reprs": sr, "similar_multihots": smh}
                    except: continue
                
                if valid_entries:
                    if self.retrieval_k == 0:
                        self.retrieval_k, self.retrieval_hidden_dim = expected_k, expected_h
                    elif expected_k != self.retrieval_k or expected_h != self.retrieval_hidden_dim:
                        logger.warning(f"Retrieval index [{rd_idx}] shape mismatch, skipping")
                        continue
                    self.retrieval_data_list.append(valid_entries)
        # Baseline-compatible expansion: one sample per non-first visit.
        # Example: patient with 4 visits contributes 3 samples (targets 2,3,4).
        self.examples = []  # list[(patient_idx, target_visit_idx)]
        for pidx, patient in enumerate(self.records):
            if hasattr(self, 'visit_level_scramble') and self.visit_level_scramble:
                # Visit-level training: unroll to all non-first visits
                for t in range(1, len(patient)):
                    self.examples.append((pidx, t))
            else:
                # Default: Only predict the final visit of each patient
                if len(patient) > 1:
                    self.examples.append((pidx, len(patient) - 1))

        # Build hadm_id → index lookups for note/lab data
        self.note_lookup = {}
        self.note_embed_dim = 768
        if note_data is not None:
            hadm_ids = note_data["hadm_ids"]
            embeddings = note_data["embeddings"]
            has_note = note_data["has_note"]
            for i, hid in enumerate(hadm_ids):
                self.note_lookup[int(hid)] = (embeddings[i], has_note[i])

        self.lab_lookup = {}
        self.lab_dim = 400
        if lab_data is not None:
            hadm_ids = lab_data["hadm_ids"]
            lab_vecs = lab_data[lab_key]
            has_lab_arr = lab_data.get("has_lab")
            self.lab_dim = lab_vecs.shape[1]
            # Phase 8: validate lab dimension logic
            if self.lab_dim < 2:
                logger.warning(f"Lab dimension {self.lab_dim} too small for flag inference.")
            for i, hid in enumerate(hadm_ids):
                has_lab = None if has_lab_arr is None else float(has_lab_arr[i])
                self.lab_lookup[int(hid)] = (lab_vecs[i], has_lab)
                
            # Store z-score stats for de-standardization in Phase 7
            if self.lab_dim in [72, 144]:
                 n_labs = self.lab_dim // 4
            else:
                 n_labs = self.lab_dim // 2
            
            self.lab_means = lab_data.get("zscore_means", np.zeros(n_labs))
            self.lab_stds = lab_data.get("zscore_stds", np.ones(n_labs))
            self.lab_names = lab_data.get("lab_names", [f"Lab_{i}" for i in range(n_labs)])

        # records format: each visit = [diag_idx, proc_idx, med_idx, hadm_id]
        # hadm_id is read directly from visit[3] during __getitem__.
        self.hadm_ids_ordered = cohort.get("hadm_ids", np.array([]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Returns a dict with all data for one patient."""
        patient_idx, target_idx = self.examples[idx]
        patient = self.records[patient_idx]

        # Prefix includes visits [0..target_idx-1], target is visit[target_idx].
        input_visits = patient[:target_idx]
        target_visit = patient[target_idx]

        # Visit sequences (history only)
        diag_seqs = []
        proc_seqs = []
        med_labels = []  # per history visit: multi-hot over num_drugs

        for visit in input_visits:
            diag_seqs.append(visit[0])  # list of diag indices
            proc_seqs.append(visit[1])  # list of proc indices
            med_vec = np.zeros(self.num_drugs, dtype=np.float32)
            for m in visit[2]:
                if m < self.num_drugs:
                    med_vec[m] = 1.0
                else:
                    raise ValueError(
                        f"Drug index {m} >= num_drugs {self.num_drugs} "
                        f"(patient {patient_idx}, history visit). "
                        f"Data preprocessing is inconsistent with num_drugs."
                    )
            med_labels.append(med_vec)

        # Target medications (current prediction visit)
        target = np.zeros(self.num_drugs, dtype=np.float32)
        for m in target_visit[2]:
            if m < self.num_drugs:
                target[m] = 1.0
            else:
                raise ValueError(
                    f"Drug index {m} >= num_drugs {self.num_drugs} "
                    f"(patient {patient_idx}, target visit). "
                    f"Data preprocessing is inconsistent with num_drugs."
                )

        # Drug history: binary OR or recency-decayed weights over history visits.
        # Run 17 Tier 1: temporal decay — drug prescribed 1 visit ago has weight 1.0,
        # 2 visits ago has weight 0.9, etc. Clip to [0,1] so it stays a soft presence signal.
        if self.use_temporal_decay and len(med_labels) > 0:
            T_hist = len(med_labels)
            drug_history = np.zeros(self.num_drugs, dtype=np.float32)
            for t, m in enumerate(med_labels):
                weight = self.decay_rate ** (T_hist - 1 - t)  # most recent = 1.0
                drug_history += weight * m
            drug_history = np.clip(drug_history, 0.0, 1.0)
        else:
            drug_history = np.clip(np.sum(med_labels, axis=0), 0, 1).astype(np.float32)

        # Per-visit medication vectors: (T, num_drugs) — used by per-visit copy mechanism
        # and medication-aware visit encoding. Unlike binary OR, preserves temporal info.
        med_per_visit = np.stack(med_labels, axis=0)  # (T, num_drugs)

        # Note/lab lookup for target visit by hadm_id in visit[3].
        note_embed = np.zeros(self.note_embed_dim, dtype=np.float32)
        has_note = 0.0
        lab_vec = np.zeros(self.lab_dim, dtype=np.float32)
        has_lab = 0.0
        
        # Phase 7: Lab delta (current - previous)
        if self.lab_dim in [72, 144]:
             n_labs = self.lab_dim // 4
        else:
             n_labs = self.lab_dim // 2

        lab_delta = np.zeros(n_labs, dtype=np.float32)
        prev_lab_vec = None

        # Run 17 Tier 1: history note aggregation — mean-pool note embeddings from
        # all history visits. Gives the model longitudinal note context.
        hist_note_embed = np.zeros(self.note_embed_dim, dtype=np.float32)
        has_hist_note = 0.0

        if self.use_history_notes:
            hist_note_vecs = []
            
        lab_trajectory = []

        for visit in input_visits:
            if len(visit) > 3:
                hid = int(visit[3])
                if self.use_history_notes and hid in self.note_lookup:
                    ne, hn = self.note_lookup[hid]
                    if float(hn) > 0.5:
                        hist_note_vecs.append(ne)
                
                # Phase 7: keep track of most recent valid lab for delta and trajectory
                if hid in self.lab_lookup:
                    lv, hl = self.lab_lookup[hid]
                    if hl is not None and float(hl) > 0.5:
                        # Also check if it's not totally missing
                        dim = len(lv)
                        n = dim // 4 if dim in [72, 144] else dim // 2
                        if not (dim >= 2 and np.all(lv[n : 2*n] > 0.5)):
                            prev_lab_vec = lv
                            if self.use_lab_trajectory:
                                lab_trajectory.append(lv)

        if self.use_history_notes and hist_note_vecs:
            hist_note_embed = np.mean(hist_note_vecs, axis=0).astype(np.float32)
            has_hist_note = 1.0

        # If records have hadm_ids (4th element), use them
        if len(target_visit) > 3:
            target_hadm = int(target_visit[3])
            if target_hadm in self.note_lookup:
                note_embed, has_note = self.note_lookup[target_hadm]
                has_note = float(has_note)
            if target_hadm in self.lab_lookup:
                lab_vec, has_lab_meta = self.lab_lookup[target_hadm]
                # FIX-CLIP-Z (B-42): clip z-score block to ±5σ before downstream
                # encoders. Single +286 σ outlier observed in MIMIC pkl can swamp
                # any Linear projection and cause gradient shocks. Flags block is
                # left untouched (binary 0/1).
                _dim = len(lab_vec)
                _n = _dim // 4 if _dim in [72, 144] else _dim // 2
                if _n > 0:
                    lab_vec = lab_vec.copy()
                    lab_vec[:_n] = np.clip(lab_vec[:_n], -5.0, 5.0)
                    # Ablation: zero out z-score block, keep missingness flags.
                    if self.lab_values_zeroed:
                        lab_vec[:_n] = 0.0
                if has_lab_meta is not None:
                    has_lab = float(has_lab_meta)
                    # BUG-006 fix: override has_lab=1 when ALL data is actually missing.
                    dim = len(lab_vec)
                    n = dim // 4 if dim in [72, 144] else dim // 2
                    if has_lab > 0 and dim >= 2:
                        missing_flags = lab_vec[n : 2*n]
                        if np.all(missing_flags > 0.5):
                            has_lab = 0.0
                elif self.lab_dim >= 2:
                    dim = len(lab_vec)
                    n = dim // 4 if dim in [72, 144] else dim // 2
                    missing_flags = lab_vec[n : 2*n]
                    has_lab = 1.0 if np.any(missing_flags < 0.5) else 0.0
                else:
                    has_lab = 1.0 if np.any(lab_vec != 0) else 0.0
                    
            # Phase 7: Compute lab delta
            if has_lab > 0 and prev_lab_vec is not None and self.lab_dim >= 2:
                dim = len(lab_vec)
                n = dim // 4 if dim in [72, 144] else dim // 2
                
                curr_missing = lab_vec[n : 2*n]
                prev_missing = prev_lab_vec[n : 2*n]
                
                curr_z = lab_vec[: n]
                prev_z = prev_lab_vec[: n]
                
                # delta = current - previous
                raw_delta = curr_z - prev_z
                
                # Mask out where either was missing
                both_present = (curr_missing < 0.5) & (prev_missing < 0.5)
                lab_delta = np.where(both_present, raw_delta, 0.0).astype(np.float32)

        # Phase 7: Dynamic lab count sweep
        if self.num_labs < n_labs:
            # Mask all components: values, flags, and (if present) trends/variances
            # Each component is of size n_labs
            num_components = self.lab_dim // n_labs
            for c in range(num_components):
                start = c * n_labs
                if c == 1: # Flags component: set to 1.0 (missing)
                    lab_vec[start + self.num_labs : start + n_labs] = 1.0
                else: # Others: set to 0.0
                    lab_vec[start + self.num_labs : start + n_labs] = 0.0
            
            # Also mask lab_delta
            if lab_delta is not None:
                lab_delta[self.num_labs:n_labs] = 0.0

        # Phase 7: HSGNN Lab Bins
        lab_bins = compute_lab_bins(lab_vec, self.lab_means, self.lab_stds, self.lab_names) if hasattr(self, "lab_means") else np.zeros(n_labs, dtype=np.int64)

        # Phase 5.4 multi-channel retrieval: lookup across all provided indices.
        num_channels = len(self.retrieval_data_list)
        similar_reprs     = np.zeros((max(num_channels, 1), self.retrieval_k, self.retrieval_hidden_dim), dtype=np.float32)
        similar_multihots = np.zeros((max(num_channels, 1), self.retrieval_k, self.num_drugs), dtype=np.float32)
        
        if num_channels > 0 and len(target_visit) > 3:
            ret_hadm = int(target_visit[3])
            for c_idx, rd in enumerate(self.retrieval_data_list):
                if ret_hadm in rd:
                    similar_reprs[c_idx]     = rd[ret_hadm]["similar_reprs"].astype(np.float32)
                    similar_multihots[c_idx] = rd[ret_hadm]["similar_multihots"].astype(np.float32)

        # Phase 4: AR sequence — prescribed drugs sorted rare-first (ascending freq).
        # If drug_freq not set, fall back to arbitrary order (by drug index).
        prescribed = np.where(target > 0.5)[0].tolist()
        if self.drug_freq is not None and len(prescribed) > 0:
            prescribed_sorted = sorted(prescribed, key=lambda d: float(self.drug_freq[d]))
        else:
            prescribed_sorted = sorted(prescribed)
        med_sequence = np.array(prescribed_sorted, dtype=np.int64)  # (n_drugs,) variable len

        sample = {
            "diag_seqs": diag_seqs,
            "proc_seqs": proc_seqs,
            "target": target,                       # (num_drugs,) multi-hot
            "drug_history": drug_history,           # (num_drugs,) decayed or binary OR
            "med_per_visit": med_per_visit,         # (T, num_drugs) per-visit med vectors
            "note_embed": note_embed,               # (768,)
            "has_note": has_note,                   # scalar
            "lab_vector": lab_vec,                  # (lab_dim,)
            "has_lab": has_lab,                     # scalar
            "lab_bins": lab_bins,                   # (n_labs,) Phase 7 clinical bins
            "lab_delta": lab_delta,                 # (n_labs,) Phase 7 lab delta
            "hist_note_embed": hist_note_embed,     # (768,) mean-pooled history notes
            "has_hist_note": has_hist_note,         # scalar
            "num_input_visits": target_idx,
            "med_sequence": med_sequence,           # (n_drugs,) AR target sequence
            "similar_reprs": similar_reprs,         # (k, H) Phase 3 retrieval
            "similar_multihots": similar_multihots, # (k, num_drugs) Phase 3 retrieval
        }

        if self.use_lab_trajectory:
            # Pad or truncate lab_trajectory to max 30
            max_traj = 30
            traj_arr = np.zeros((max_traj, self.lab_dim), dtype=np.float32)
            traj_len = min(len(lab_trajectory), max_traj)
            if traj_len > 0:
                # Take the most recent `traj_len` visits
                recent_traj = np.array(lab_trajectory[-traj_len:], dtype=np.float32)
                traj_arr[:traj_len] = recent_traj
            sample["lab_trajectory"] = traj_arr
            sample["lab_trajectory_len"] = traj_len

        return sample


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate: pad variable-length visit sequences and code lists.

    Returns tensors ready for the model.
    """
    batch_size = len(batch)
    max_visits = max(b["num_input_visits"] for b in batch)

    # Find max codes per visit position
    max_diag_per_visit = []
    max_proc_per_visit = []
    for t in range(max_visits):
        md = max(
            (len(b["diag_seqs"][t]) if t < b["num_input_visits"] else 0)
            for b in batch
        )
        mp = max(
            (len(b["proc_seqs"][t]) if t < b["num_input_visits"] else 0)
            for b in batch
        )
        max_diag_per_visit.append(max(md, 1))  # at least 1 to avoid empty tensors
        max_proc_per_visit.append(max(mp, 1))

    # Build padded sequences
    diag_seq_tensors = []
    proc_seq_tensors = []
    diag_mask_tensors = []
    proc_mask_tensors = []

    for t in range(max_visits):
        md = max_diag_per_visit[t]
        mp = max_proc_per_visit[t]

        diag_t = torch.zeros(batch_size, md, dtype=torch.long)
        proc_t = torch.zeros(batch_size, mp, dtype=torch.long)
        dmask_t = torch.zeros(batch_size, md, dtype=torch.bool)
        pmask_t = torch.zeros(batch_size, mp, dtype=torch.bool)

        for i, b in enumerate(batch):
            if t < b["num_input_visits"]:
                d = b["diag_seqs"][t]
                p = b["proc_seqs"][t]
                if d:
                    diag_t[i, :len(d)] = torch.tensor(d, dtype=torch.long)
                    dmask_t[i, :len(d)] = True
                if p:
                    proc_t[i, :len(p)] = torch.tensor(p, dtype=torch.long)
                    pmask_t[i, :len(p)] = True

        diag_seq_tensors.append(diag_t)
        proc_seq_tensors.append(proc_t)
        diag_mask_tensors.append(dmask_t)
        proc_mask_tensors.append(pmask_t)

    lengths = torch.tensor([b["num_input_visits"] for b in batch], dtype=torch.long)
    targets = torch.tensor(np.stack([b["target"] for b in batch]), dtype=torch.float32)
    drug_history = torch.tensor(np.stack([b["drug_history"] for b in batch]), dtype=torch.float32)
    note_embed = torch.tensor(np.stack([b["note_embed"] for b in batch]), dtype=torch.float32)
    has_note = torch.tensor([b["has_note"] for b in batch], dtype=torch.float32)
    lab_vector = torch.tensor(np.stack([b["lab_vector"] for b in batch]), dtype=torch.float32)
    has_lab = torch.tensor([b["has_lab"] for b in batch], dtype=torch.float32)
    lab_bins = torch.tensor(np.stack([b["lab_bins"] for b in batch]), dtype=torch.long)
    lab_delta = torch.tensor(np.stack([b["lab_delta"] for b in batch]), dtype=torch.float32)
    hist_note_embed = torch.tensor(
        np.stack([b["hist_note_embed"] for b in batch]), dtype=torch.float32
    )
    has_hist_note = torch.tensor([b["has_hist_note"] for b in batch], dtype=torch.float32)

    # Per-visit medication vectors: pad to (batch, max_visits, num_drugs)
    num_drugs = batch[0]["target"].shape[0]
    med_per_visit = torch.zeros(batch_size, max_visits, num_drugs, dtype=torch.float32)
    for i, b in enumerate(batch):
        T = b["num_input_visits"]
        med_per_visit[i, :T] = torch.from_numpy(b["med_per_visit"][:T])

    # Phase 5.4: Multi-channel Retrieval tensors — (batch, C, k, H) and (batch, C, k, num_drugs)
    if batch[0]["similar_reprs"].shape[1] > 0:
        similar_reprs     = torch.tensor(
            np.stack([b["similar_reprs"]     for b in batch]), dtype=torch.float32
        )  # (batch, C, k, H)
        similar_multihots = torch.tensor(
            np.stack([b["similar_multihots"] for b in batch]), dtype=torch.float32
        )  # (batch, C, k, num_drugs)
    else:
        similar_reprs     = None
        similar_multihots = None

    # Phase 4: AR med_sequence — pad to max_seq_len with -1 (ignored in loss)
    max_seq = max(len(b["med_sequence"]) for b in batch)
    max_seq = max(max_seq, 1)
    med_sequence = torch.full((batch_size, max_seq), -1, dtype=torch.long)
    seq_lengths  = torch.zeros(batch_size, dtype=torch.long)
    for i, b in enumerate(batch):
        seq = b["med_sequence"]
        n   = len(seq)
        if n > 0:
            med_sequence[i, :n] = torch.tensor(seq, dtype=torch.long)
        seq_lengths[i] = n

    ret = {
        "diag_seq": diag_seq_tensors,
        "proc_seq": proc_seq_tensors,
        "diag_mask_seq": diag_mask_tensors,
        "proc_mask_seq": proc_mask_tensors,
        "lengths": lengths,
        "target": targets,
        "drug_history": drug_history,
        "med_per_visit": med_per_visit,
        "note_embed": note_embed,
        "has_note": has_note,
        "lab_vector": lab_vector,
        "has_lab": has_lab,
        "lab_bins": lab_bins,
        "lab_delta": lab_delta,
        "hist_note_embed": hist_note_embed,
        "has_hist_note": has_hist_note,
        "med_sequence": med_sequence,
        "seq_lengths": seq_lengths,
        "similar_reprs": similar_reprs,
        "similar_multihots": similar_multihots,
    }

    if "lab_trajectory" in batch[0]:
        ret["lab_trajectory"] = torch.tensor(np.stack([b["lab_trajectory"] for b in batch]), dtype=torch.float32)
        ret["lab_trajectory_len"] = torch.tensor([b["lab_trajectory_len"] for b in batch], dtype=torch.long)

    return ret
