# src/eval/cost_threshold.py
from __future__ import annotations

from typing import Any, Dict, List


def best_by_cost(
    rows: List[Dict[str, Any]],
    *,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> Dict[str, Any]:
    """
    Select threshold that minimizes: cost = cost_fp*FP + cost_fn*FN

    Expects rows from sweep_thresholds() which include tp/fp/tn/fn + precision/recall/f1/threshold.
    Returns the best row plus 'cost'.
    """
    if not rows:
        raise ValueError("rows is empty")

    if cost_fp <= 0 or cost_fn <= 0:
        raise ValueError("cost_fp and cost_fn must be > 0")

    best = None
    best_cost = float("inf")

    for r in rows:
        fp = float(r.get("fp", 0))
        fn = float(r.get("fn", 0))
        c = cost_fp * fp + cost_fn * fn

        if c < best_cost:
            best_cost = c
            best = r
        elif c == best_cost and best is not None:
            # Tie-breakers: prefer higher recall, then higher F1, then lower threshold
            if float(r.get("recall", 0.0)) > float(best.get("recall", 0.0)):
                best = r
            elif float(r.get("recall", 0.0)) == float(best.get("recall", 0.0)):
                if float(r.get("f1", 0.0)) > float(best.get("f1", 0.0)):
                    best = r
                elif float(r.get("f1", 0.0)) == float(best.get("f1", 0.0)):
                    if float(r.get("threshold", 1.0)) < float(best.get("threshold", 1.0)):
                        best = r

    out = dict(best)
    out["cost_fp"] = float(cost_fp)
    out["cost_fn"] = float(cost_fn)
    out["cost"] = float(best_cost)
    return out