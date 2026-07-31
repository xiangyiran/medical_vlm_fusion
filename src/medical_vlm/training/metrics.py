"""Original multi-label evaluation metrics."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ..config import LABEL_COLS

def safe_macro_auroc(y_true, y_prob):
    """Compute macro AUROC while skipping labels that have only one class in the split."""
    scores, per_class = [], {}
    for i, c in enumerate(LABEL_COLS):
        if len(np.unique(y_true[:, i])) < 2:
            per_class[c] = np.nan
            continue
        try:
            score = roc_auc_score(y_true[:, i], y_prob[:, i])
            scores.append(score)
            per_class[c] = float(score)
        except Exception:
            per_class[c] = np.nan
    return float(np.mean(scores)) if scores else np.nan, per_class

def safe_macro_ap(y_true, y_prob):
    """Compute macro average precision while skipping invalid labels."""
    scores, per_class = [], {}
    for i, c in enumerate(LABEL_COLS):
        if len(np.unique(y_true[:, i])) < 2:
            per_class[c] = np.nan
            continue
        try:
            score = average_precision_score(y_true[:, i], y_prob[:, i])
            scores.append(score)
            per_class[c] = float(score)
        except Exception:
            per_class[c] = np.nan
    return float(np.mean(scores)) if scores else np.nan, per_class

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute classification metrics for multi-label disease prediction."""
    y_pred = (y_prob >= threshold).astype(np.float32)
    macro_auroc, per_label_auroc = safe_macro_auroc(y_true, y_prob)
    macro_ap, per_label_ap = safe_macro_ap(y_true, y_prob)
    return {
        "macro_auroc": macro_auroc,
        "macro_ap": macro_ap,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "label_accuracy": float((y_true == y_pred).mean()),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "per_label_auroc": per_label_auroc,
        "per_label_ap": per_label_ap,
    }
