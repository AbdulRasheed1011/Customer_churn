from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    out: List[Dict[str, Any]] = []
    y_true = y_true.astype(int)

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        out.append(
            {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    return out


def best_by_f1(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("No threshold rows to select from.")
    return max(rows, key=lambda r: r["f1"])