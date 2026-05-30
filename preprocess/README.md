# Preprocess Contract (MIMIC-III)

## Overview
This folder contains the preprocessing contracts that convert raw MIMIC tables into model-ready artifacts. These outputs are treated as stable interfaces by training and evaluation.

## Inputs
Required raw MIMIC-III tables:
- ADMISSIONS.csv.gz
- PRESCRIPTIONS.csv.gz
- DIAGNOSES_ICD.csv.gz
- PROCEDURES_ICD.csv.gz
- NOTEEVENTS.csv.gz (for notes extraction)
- LABEVENTS.csv.gz (for lab extraction)

Required external mapping files under data/external:
- ndc2RXCUI.txt
- RXCUI2atc4.csv
- idx2SMILES.pkl
- drug-DDI.csv
- (optional but recommended) ddi_A_final_carmen.pkl + voc_carmen_mimic3.pkl

## Outputs
Primary structured artifacts in data/processed:
- records_final.pkl: patient records, each visit = [diag_idx, proc_idx, med_idx, hadm_id]
- voc_final.pkl: diag/proc/med vocabularies
- ddi_A_final.pkl: drug-drug interaction adjacency matrix
- ehr_adj_final.pkl: drug co-occurrence adjacency matrix (train split only)
- cohort_mimic3.pkl: metadata with num_drugs/num_diag/num_proc, hadm_ids, split_indices, split_seed
- preprocess_manifest.json: reproducibility metadata (counts, parameters, DDI density)

Optional modality artifacts:
- notes_text_mimic3.pkl / note_embeddings_mimic3.pkl
- lab_data_mimic3.pkl

## Guarantees
- Records schema is fixed at 4 fields per visit.
- hadm_id is preserved for notes/labs alignment.
- Split indices are patient-level and deterministic for a given seed.
- DDI and EHR matrices are square with side length = num_drugs.
- Manifest captures run settings and key cohort statistics.

## Leakage Guards
- Co-occurrence graph is built from training records only.
- Split metadata is persisted and should be reused by training.
- Notes cleaning removes discharge-medication sections and reports residual markers.
- Lab z-score statistics are computed from training split only.

## Quick Commands (MIMIC-III)
Preprocess:
python src/preprocess/preprocess_mimic3.py --mimic_dir DATASETS/mimic-iii-clinical-database-1.4 --external_dir data/external --output_dir data/processed --preprocess_mode canonical --min_visits 2 --final_min_visits 1

Extract notes:
python src/preprocess/extract_notes.py --mimic_dir DATASETS/mimic-iii-clinical-database-1.4 --cohort_file data/processed/cohort_mimic3.pkl --output_dir data/processed --mimic_version 3

Extract labs:
python src/preprocess/extract_labs.py --mimic_dir DATASETS/mimic-iii-clinical-database-1.4 --cohort_file data/processed/cohort_mimic3.pkl --output_dir data/processed --mimic_version 3

Run lockdown audit:
python scripts/preprocess_lockdown_audit.py --processed_dir data/processed --mimic_version 3 --strict

## Troubleshooting Pointers
- Missing file errors: validate file tags (final/mimic3) and paths.
- Split mismatch: inspect cohort_mimic3.pkl split_indices and split_seed.
- Shape mismatch: compare cohort num_drugs against ddi/ehr matrix shapes.
- Modality alignment issues: check hadm overlap fields in lockdown_audit_report.json.
