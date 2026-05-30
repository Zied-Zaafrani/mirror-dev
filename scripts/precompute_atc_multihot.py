import pickle
import numpy as np
import os
import sys

from collections import defaultdict

def resolve_drug_vocab(cohort: dict):
    # Simplified copy of resolve_drug_vocab
    for key in ("drug_vocab", "med_voc"):
        vocab = cohort.get(key)
        if not isinstance(vocab, dict): continue
        if isinstance(vocab.get("idx2word"), dict):
            return vocab["idx2word"]
        inverted = {v: k for k, v in vocab.items() if isinstance(k, str) and isinstance(v, int)}
        if inverted: return inverted
        idx2word = {}
        for k, v in vocab.items():
            if isinstance(v, str):
                try: idx2word[int(k)] = v
                except: pass
        if idx2word: return idx2word
    return None

def main():
    data_dir = "data/processed"
    voc_path = os.path.join(data_dir, "voc_final.pkl")
    with open(voc_path, "rb") as f:
        cohort = pickle.load(f)
    
    idx2word = resolve_drug_vocab(cohort)
    if not idx2word:
        print("ERROR: Could not resolve drug vocab")
        return

    num_drugs = max(idx2word.keys()) + 1
    print(f"Num drugs: {num_drugs}")

    # Build ATC-3 classes
    atc3_groups = defaultdict(list)
    drug_names = {}
    for idx in range(num_drugs):
        name = idx2word.get(idx, idx2word.get(str(idx), ""))
        drug_names[idx] = name if isinstance(name, str) else ""
        if isinstance(name, str) and len(name) >= 3:
            atc3_groups[name[:3]].append(idx)
    
    singleton_drugs = set()
    for group_key, group_drugs in atc3_groups.items():
        if len(group_drugs) == 1:
            singleton_drugs.add(group_drugs[0])

    atc2_groups = defaultdict(list)
    for idx in singleton_drugs:
        name = drug_names.get(idx, "")
        if len(name) >= 2:
            atc2_groups[name[:2]].append(idx)

    # Assign each drug to an ATC class ID
    # Classes are unique prefixes (ATC-3 or ATC-2 fallback)
    all_classes = set(atc3_groups.keys()) | set(atc2_groups.keys())
    sorted_classes = sorted(list(all_classes))
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    num_atc_classes = len(class_to_idx)

    drug_to_atc3 = {} # drug_idx -> list of class indices
    for idx in range(num_drugs):
        drug_to_atc3[idx] = []
        name = drug_names.get(idx, "")
        if len(name) >= 3:
            if idx in singleton_drugs and len(name) >= 2:
                drug_to_atc3[idx].append(class_to_idx[name[:2]])
            else:
                drug_to_atc3[idx].append(class_to_idx[name[:3]])

    # Coverage diagnostic
    covered = sum(1 for idx in range(num_drugs) if len(drug_to_atc3[idx]) > 0)
    print(f"Diagnostic: num_atc_classes={num_atc_classes}, coverage={covered/num_drugs*100:.2f}%")

    # Compute multihot per visit
    records_path = os.path.join(data_dir, "records_final.pkl")
    with open(records_path, "rb") as f:
        records = pickle.load(f)

    # Instead of full patient ordering, we just save a projection matrix!
    # A simple matrix (num_drugs, num_atc_classes)
    # Then in collate_fn: atc_multihot = (target_multihot @ projection_matrix) > 0
    projection_matrix = np.zeros((num_drugs, num_atc_classes), dtype=np.float32)
    for drug_idx, classes in drug_to_atc3.items():
        for c in classes:
            projection_matrix[drug_idx, c] = 1.0

    np.save(os.path.join(data_dir, "drug_to_atc_projection.npy"), projection_matrix)
    print("Saved data/processed/drug_to_atc_projection.npy")

if __name__ == "__main__":
    main()
