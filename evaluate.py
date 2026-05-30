"""
Evaluation metrics for drug recommendation.

Standard metrics: Jaccard, F1, PRAUC, DDI Rate.
All computed per-visit then averaged across test set.

Verified against COGNet/SafeDrug baselines:
- Jaccard: identical formula (set intersection/union, per-visit average)
- F1: positive-class-only (binary F1), more meaningful than baselines' macro F1
  NOTE: SafeDrug/COGNet report F1 with sklearn average='macro' which includes
  correct non-prescriptions and gives ~5-8% higher F1 numbers. Our F1 is lower
  but more honest — we only measure prescription accuracy, not "correctly not giving drugs."
  For thesis comparisons: Jaccard is the primary metric (identical across papers).
- PRAUC: identical (sklearn average_precision_score per-visit)
- DDI Rate: identical (fraction of predicted pairs with known interaction)

Extra diagnostics (added for fusion debugging):
- Precision + Recall separately — tells us if fusion causes over/under-prediction
- Avg True Meds + Avg Pred Meds — tells us if model is predict too many/few drugs
- evaluate_threshold_sweep — finds if model is systematically over/underconfident
"""

import numpy as np
from sklearn.metrics import average_precision_score


def jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-sample Jaccard similarity, averaged over batch.

    Verified identical to COGNet/SafeDrug set-based formula.
    """
    scores = []
    for yt, yp in zip(y_true, y_pred):
        inter = np.sum(yt * yp)
        union = np.sum(np.clip(yt + yp, 0, 1))
        scores.append(inter / max(union, 1e-8))
    return float(np.mean(scores))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-sample F1 (positive-class only), averaged over batch.

    METHODOLOGY NOTE: This is equivalent to sklearn f1_score(average='binary').
    SafeDrug/COGNet use f1_score(average='macro') which also rewards correctly
    NOT prescribing drugs → inflated F1 (typically +5-8%). Our formula only
    measures prescription accuracy (TP / predicted, TP / true). More meaningful
    for drug recommendation. Jaccard is the primary comparison metric.
    """
    scores = []
    for yt, yp in zip(y_true, y_pred):
        tp = np.sum(yt * yp)
        prec = tp / max(np.sum(yp), 1e-8)
        rec = tp / max(np.sum(yt), 1e-8)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        scores.append(f1)
    return float(np.mean(scores))


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Per-sample Precision and Recall separately, averaged over batch.

    Returns (precision, recall).

    Why both separately matters:
      - High precision, low recall → under-prescription (too cautious)
      - Low precision, high recall → over-prescription (predicting too many drugs)
    When notes/labs hurt Jaccard, checking precision vs recall tells us WHY:
      - If precision drops: fusion is adding noise → model predicts more drugs
      - If recall drops: fusion is suppressing signal → model predicts fewer drugs
    """
    precs, recs = [], []
    for yt, yp in zip(y_true, y_pred):
        tp = np.sum(yt * yp)
        precs.append(tp / max(np.sum(yp), 1e-8))
        recs.append(tp / max(np.sum(yt), 1e-8))
    return float(np.mean(precs)), float(np.mean(recs))


def prauc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Per-sample PRAUC (area under precision-recall curve), averaged."""
    scores = []
    for yt, yp in zip(y_true, y_prob):
        if yt.sum() == 0:
            continue
        try:
            scores.append(average_precision_score(yt, yp))
        except ValueError as e:
            import logging
            logging.getLogger(__name__).debug(f"PRAUC skipped for sample: {e}")
            continue
    return float(np.mean(scores)) if scores else 0.0


def ddi_rate(y_pred: np.ndarray, ddi_adj: np.ndarray) -> float:
    """Fraction of predicted drug pairs with known DDI interaction.

    Verified identical to COGNet/SafeDrug formula:
      sum of ddi pairs across all visits / sum of total pairs across all visits.

    Args:
        y_pred: (batch, num_drugs) binary predictions
        ddi_adj: (num_drugs, num_drugs) binary DDI matrix

    Returns:
        DDI rate (lower is better)
    """
    total_pairs = 0
    ddi_pairs = 0
    for yp in y_pred:
        drugs = np.where(yp == 1)[0]
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                total_pairs += 1
                if ddi_adj[drugs[i], drugs[j]] == 1:
                    ddi_pairs += 1
    return ddi_pairs / max(total_pairs, 1e-8)


def avg_med_count(y_pred: np.ndarray) -> float:
    """Average number of predicted medications per visit."""
    return float(np.mean(y_pred.sum(axis=1)))


def avg_true_med_count(y_true: np.ndarray) -> float:
    """Average number of ground-truth medications per visit."""
    return float(np.mean(y_true.sum(axis=1)))


def non_historical_jaccard(y_true: np.ndarray, y_pred: np.ndarray, med_per_visit: np.ndarray, lengths: np.ndarray) -> tuple[float, float]:
    """Calculate Jaccard isolated to actual medication changes.
    
    To combat the "Memory Cheat" where models achieve high Jaccard simply by 
    copying the previous visit, we measure:
    1. NHJ_new: Accuracy on drugs newly added.
    2. NHJ_dropped: Accuracy on drugs that were stopped.
    
    Args:
        y_true: (N, num_drugs) binary ground truth for current visit
        y_pred: (N, num_drugs) binary predictions for current visit
        med_per_visit: (N, max_visits, num_drugs) binary history
        lengths: (N,) number of input visits per patient
        
    Returns:
        (nhj_new, nhj_dropped)
    """
    batch_size, _, num_drugs = med_per_visit.shape
    H = np.zeros_like(y_true)
    for i in range(batch_size):
        # lengths[i] is the number of history visits. The most recent is lengths[i] - 1
        t = int(lengths[i]) - 1
        if t >= 0:
            H[i] = med_per_visit[i, t]

    # Target changes
    T_new = np.clip(y_true - H, 0, 1)
    P_new = np.clip(y_pred - H, 0, 1)

    T_dropped = np.clip(H - y_true, 0, 1)
    P_dropped = np.clip(H - y_pred, 0, 1)

    scores_new = []
    scores_dropped = []

    for tn, pn, td, pd in zip(T_new, P_new, T_dropped, P_dropped):
        union_n = np.sum(np.clip(tn + pn, 0, 1))
        if union_n > 0:
            inter_n = np.sum(tn * pn)
            scores_new.append(inter_n / union_n)

        union_d = np.sum(np.clip(td + pd, 0, 1))
        if union_d > 0:
            inter_d = np.sum(td * pd)
            scores_dropped.append(inter_d / union_d)

    nhj_new = float(np.mean(scores_new)) if scores_new else 0.0
    nhj_dropped = float(np.mean(scores_dropped)) if scores_dropped else 0.0
        
    return nhj_new, nhj_dropped


def evaluate_all(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    y_pred_prob: np.ndarray,
    ddi_adj: np.ndarray,
    med_per_visit: np.ndarray | None = None,
    lengths: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all standard metrics plus diagnostic metrics.

    Args:
        y_true: (N, num_drugs) binary ground truth
        y_pred_binary: (N, num_drugs) binary predictions (after threshold)
        y_pred_prob: (N, num_drugs) predicted probabilities
        ddi_adj: (num_drugs, num_drugs) binary DDI matrix
        med_per_visit: (Optional) (N, max_visits, num_drugs) binary history for NHJ
        lengths: (Optional) (N,) number of input visits per patient for NHJ

    Returns:
        dict with standard metrics (Jaccard, F1, PRAUC, DDI Rate, Avg Meds)
        plus diagnostic metrics (Precision, Recall, Avg True Meds, NHJ_new, NHJ_dropped).
    """
    prec, rec = precision_recall(y_true, y_pred_binary)
    metrics = {
        # Primary comparison metric
        "Jaccard": jaccard(y_true, y_pred_binary),
        # Standard reported metrics
        "F1": f1_score(y_true, y_pred_binary),
        "PRAUC": prauc(y_true, y_pred_prob),
        "DDI Rate": ddi_rate(y_pred_binary, ddi_adj),
        # Diagnostic: prescription count
        "Avg Meds": avg_med_count(y_pred_binary),
        "Avg True Meds": avg_true_med_count(y_true),
        "Precision": prec,
        "Recall": rec,
    }

    if med_per_visit is not None and lengths is not None:
        nhj_new, nhj_dropped = non_historical_jaccard(y_true, y_pred_binary, med_per_visit, lengths)
        metrics["NHJ_new"] = nhj_new
        metrics["NHJ_dropped"] = nhj_dropped
    else:
        metrics["NHJ_new"] = 0.0
        metrics["NHJ_dropped"] = 0.0

    return metrics


def compute_drug_frequency_tiers(
    y_true_train: np.ndarray,
    tier_thresholds: tuple[float, float] = (0.10, 0.40),
) -> dict[str, np.ndarray]:
    """Classify drugs into ubiquitous/moderate/rare tiers by training frequency.

    Args:
        y_true_train: (N_train, num_drugs) binary ground truth from training set.
        tier_thresholds: (rare_upper, ubiq_lower) — fraction of visits.
            Drugs appearing in <rare_upper fraction of visits → RARE.
            Drugs appearing in >ubiq_lower fraction of visits → UBIQUITOUS.
            Everything else → MODERATE.

    Returns:
        dict with 'rare', 'moderate', 'ubiquitous' keys mapping to drug index arrays,
        plus 'frequencies' (per-drug prescription fraction) for analysis.
    """
    n_visits = y_true_train.shape[0]
    freq = y_true_train.sum(axis=0) / max(n_visits, 1)  # (num_drugs,)
    rare_thresh, ubiq_thresh = tier_thresholds

    rare_mask = freq < rare_thresh
    ubiq_mask = freq >= ubiq_thresh
    moderate_mask = ~rare_mask & ~ubiq_mask

    return {
        "rare": np.where(rare_mask)[0],
        "moderate": np.where(moderate_mask)[0],
        "ubiquitous": np.where(ubiq_mask)[0],
        "frequencies": freq,
        "tier_thresholds": tier_thresholds,
        "counts": {
            "rare": int(rare_mask.sum()),
            "moderate": int(moderate_mask.sum()),
            "ubiquitous": int(ubiq_mask.sum()),
        },
    }


def jaccard_stratified(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tiers: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute per-tier Jaccard — the thesis-critical diagnostic.

    Answers: does the multimodal delta come from ubiquitous drugs
    (worthless for thesis) or rare drugs (thesis-grade)?

    Args:
        y_true: (N, num_drugs) binary ground truth
        y_pred: (N, num_drugs) binary predictions
        tiers: output of compute_drug_frequency_tiers()

    Returns:
        dict with Jaccard per tier + overall, e.g.:
        {"Jaccard_rare": 0.12, "Jaccard_moderate": 0.45, "Jaccard_ubiquitous": 0.71,
         "n_rare": 39, "n_moderate": 52, "n_ubiquitous": 39}
    """
    result = {}
    for tier_name in ("rare", "moderate", "ubiquitous"):
        drug_indices = tiers[tier_name]
        if len(drug_indices) == 0:
            result[f"Jaccard_{tier_name}"] = 0.0
            result[f"n_{tier_name}"] = 0
            continue
        yt_tier = y_true[:, drug_indices]
        yp_tier = y_pred[:, drug_indices]
        # Per-visit Jaccard on this tier's drugs only
        scores = []
        for yt, yp in zip(yt_tier, yp_tier):
            inter = np.sum(yt * yp)
            union = np.sum(np.clip(yt + yp, 0, 1))
            if union > 0:  # Skip visits with no drugs in this tier
                scores.append(inter / union)
        result[f"Jaccard_{tier_name}"] = float(np.mean(scores)) if scores else 0.0
        result[f"n_{tier_name}"] = len(drug_indices)
    return result


def compute_per_drug_thresholds(
    y_prob_val: np.ndarray,
    y_true_val: np.ndarray,
    thresholds: list[float] | None = None,
) -> np.ndarray:
    """Compute per-drug optimal threshold from validation set.

    For each drug d, sweep candidate thresholds and pick the one that maximises
    F1 for that drug independently.  This is the "@pd" (per-drug threshold) mode
    used in Run 17+ best results.

    IMPORTANT: Must be fitted on the VALIDATION set only, then applied to TEST.
    Fitting on test data would be data leakage.

    Args:
        y_prob_val:  (N_val, num_drugs) sigmoid probabilities from validation set
        y_true_val:  (N_val, num_drugs) binary ground truth for validation set
        thresholds:  candidate thresholds (default: 50 values from 0.05 to 0.95)

    Returns:
        per_drug_thresholds: (num_drugs,) optimal threshold per drug
    """
    if thresholds is None:
        thresholds = list(np.linspace(0.05, 0.95, 50))

    num_drugs = y_prob_val.shape[1]
    per_drug_thresh = np.full(num_drugs, 0.5, dtype=np.float32)

    for d in range(num_drugs):
        yt = y_true_val[:, d]
        yp = y_prob_val[:, d]
        if yt.sum() == 0:
            continue  # drug never prescribed in val → keep 0.5
        best_f1, best_t = -1.0, 0.5
        for t in thresholds:
            yb = (yp >= t).astype(np.float32)
            tp = np.sum(yt * yb)
            prec = tp / max(np.sum(yb), 1e-8)
            rec  = tp / max(np.sum(yt), 1e-8)
            f1   = 2 * prec * rec / max(prec + rec, 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        per_drug_thresh[d] = best_t

    return per_drug_thresh


def apply_per_drug_thresholds(
    y_prob: np.ndarray,
    per_drug_thresholds: np.ndarray,
) -> np.ndarray:
    """Apply per-drug thresholds to convert probabilities to binary predictions.

    Args:
        y_prob:              (N, num_drugs) predicted probabilities
        per_drug_thresholds: (num_drugs,)   threshold per drug (from compute_per_drug_thresholds)

    Returns:
        y_binary: (N, num_drugs) binary predictions
    """
    # Vectorised comparison: broadcast thresholds across samples
    return (y_prob >= per_drug_thresholds[np.newaxis, :]).astype(np.float32)


def evaluate_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ddi_adj: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[float, dict[str, float]]:
    """Sweep threshold values and report Jaccard, avg meds, DDI rate for each.

    Useful at test time to check if the model is systematically over/underconfident:
    - If peak Jaccard is at threshold < 0.5: model is underconfident (logits too low)
      → sigmoid < 0.5 for drugs it should predict → lower threshold recovers them
    - If peak Jaccard is at threshold > 0.5: model is overconfident (predicts too many)
      → raise threshold to be more selective

    Baselines all use 0.5 (marked with ← in output). For fair comparison, always
    report final numbers at threshold=0.5.

    Args:
        y_true: (N, num_drugs) binary ground truth
        y_prob: (N, num_drugs) predicted probabilities
        ddi_adj: (num_drugs, num_drugs) binary DDI matrix
        thresholds: list of thresholds to try (default: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    Returns:
        dict mapping threshold → {Jaccard, Avg Meds, Avg True Meds, DDI Rate}
    """
    if thresholds is None:
        thresholds = [round(i * 0.05, 2) for i in range(8, 16)]  # 0.40 … 0.75

    results = {}
    for t in thresholds:
        y_bin = apply_threshold(y_prob, threshold=t, top_k=None)
        results[t] = {
            "Jaccard": jaccard(y_true, y_bin),
            "Avg Meds": avg_med_count(y_bin),
            "Avg True Meds": avg_true_med_count(y_true),
            "DDI Rate": ddi_rate(y_bin, ddi_adj),
        }
    return results


def apply_threshold(
    y_prob: np.ndarray,
    threshold: float = 0.5,
    top_k: int | None = None,
) -> np.ndarray:
    """Apply threshold to convert probabilities to binary predictions.

    Default: simple 0.5 threshold (matches SafeDrug/GAMENet/all baselines).
    Optional: top-K pre-filter before thresholding.

    Args:
        y_prob: (N, num_drugs) predicted probabilities
        threshold: probability threshold (default 0.5)
        top_k: if set, only consider top-K drugs before applying threshold

    Returns:
        y_binary: (N, num_drugs) binary predictions
    """
    batch_size, num_drugs = y_prob.shape
    y_binary = np.zeros_like(y_prob, dtype=np.float32)

    for i in range(batch_size):
        probs = y_prob[i]

        if top_k is not None:
            # Top-K selection: pick the top_k highest-probability drugs.
            # If threshold is also provided, further filter by it.
            # If threshold is None, select all top_k unconditionally.
            topk_idx = np.argsort(probs)[-top_k:]
            if threshold is None:
                selected = topk_idx
            else:
                selected = topk_idx[probs[topk_idx] >= threshold]
        else:
            # Simple threshold (standard baseline approach)
            selected = np.where(probs >= threshold)[0]

        # FIX-EVAL-002: Removed forced-argmax fallback.
        # Old code forced np.argmax(probs) when all sigmoid < threshold,
        # preventing Jaccard from ever hitting 0 for hard patients.
        # Baselines (SafeDrug, COGNet, GAMENet) allow empty predictions → Jaccard=0.
        # Keeping the fallback caused +0.003–0.01 artificial floor inflation.
        # For clinical inference (not metric eval), use apply_threshold_inference() instead.

        y_binary[i, selected] = 1.0

    return y_binary


def evaluate_by_visit_count(y_true: np.ndarray, y_pred: np.ndarray, lengths: np.ndarray) -> dict[str, float]:
    """Split evaluation by single-visit vs multi-visit.
    
    Args:
        lengths: (N,) — visit count per patient
    """
    single = lengths == 1
    multi = lengths > 1
    
    results = {
        "Jaccard_all": jaccard(y_true, y_pred),
    }
    
    if single.sum() > 0:
        results["Jaccard_single_visit"] = jaccard(y_true[single], y_pred[single])
        results["single_visit_pct"] = float(single.sum() / len(single))
    else:
        results["Jaccard_single_visit"] = 0.0
        results["single_visit_pct"] = 0.0
        
    if multi.sum() > 0:
        results["Jaccard_multi_visit"] = jaccard(y_true[multi], y_pred[multi])
        results["multi_visit_pct"] = float(multi.sum() / len(multi))
    else:
        results["Jaccard_multi_visit"] = 0.0
        results["multi_visit_pct"] = 0.0
        
    return results
