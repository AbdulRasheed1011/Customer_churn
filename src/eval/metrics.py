# src/eval/metrics.py
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def classification_metrics_at_threshold(
    y_true_bin: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float,
    positive_label: int = 1,
) -> Dict[str, Any]:
    """
    Canonical 4-metrics computation (accuracy/precision/recall/f1) at a given threshold.

    y_true_bin: array of 0/1 ground truth
    y_proba:    array of predicted probabilities for class 1
    threshold:  decision threshold in [0,1]
    """
    if y_true_bin is None or y_proba is None:
        raise ValueError("y_true_bin and y_proba must not be None")

    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    if y_true_bin.shape[0] != y_proba.shape[0]:
        raise ValueError(f"Shape mismatch: y_true={y_true_bin.shape}, y_proba={y_proba.shape}")

    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"threshold must be in [0,1]. Got {threshold}")

    y_pred = (y_proba >= float(threshold)).astype(int)

    # IMPORTANT: zero_division=0 avoids crashing when model predicts no positives
    prec = float(precision_score(y_true_bin, y_pred, pos_label=positive_label, zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred, pos_label=positive_label, zero_division=0))
    f1 = float(f1_score(y_true_bin, y_pred, pos_label=positive_label, zero_division=0))
    acc = float(accuracy_score(y_true_bin, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }