"""Classification, calibration and worst-group metrics on tumour probabilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


def fast_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney, ties get the average rank); NaN when only one class is present."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(probs, method="average")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> tuple[float, float, list[int]]:
    """Expected and maximum calibration error over equal-width bins, plus the bin counts."""
    bounds = np.linspace(0, 1, n_bins + 1)
    ece, mce, counts = 0.0, 0.0, []
    for b in range(n_bins):
        upper = (probs < bounds[b + 1]) if b < n_bins - 1 else (probs <= bounds[b + 1])
        in_bin = (probs >= bounds[b]) & upper
        counts.append(int(in_bin.sum()))
        if in_bin.sum():
            gap = abs(probs[in_bin].mean() - labels[in_bin].mean())
            ece += in_bin.sum() * gap
            mce = max(mce, gap)
    return float(ece / len(labels)), float(mce), counts


def metrics_from(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """Full metric set at threshold 0.5 (keys identical to the legacy result files)."""
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    two_classes = len(np.unique(labels)) > 1
    ece, mce, counts = calibration(probs, labels, n_bins)
    return {
        "n": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "auc": float(roc_auc_score(labels, probs)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(labels, probs)) if two_classes else float("nan"),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "wga": float(min(sens, spec)),
        "brier": float(brier_score_loss(labels, probs)),
        "ece": ece,
        "mce": mce,
        "bin_counts": counts,
        "cost_error_10to1": float((10 * fn + fp) / len(labels)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
