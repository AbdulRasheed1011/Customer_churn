# src/models/catboost_three_way.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.eval.threshold import sweep_thresholds, best_by_f1
from src.eval.cost_threshold import best_by_cost


@dataclass
class CatBoostThreeWayResult:
    model: CatBoostClassifier
    best_threshold_val: Dict[str, Any]
    metrics_test_locked: Dict[str, Any]
    y_val_proba: np.ndarray
    y_test_proba: np.ndarray
    cat_feature_names: List[str]


def _infer_cat_features(X: pd.DataFrame) -> List[str]:
    # CatBoost handles strings/categories directly; treat object/category/bool as categorical.
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return cat_cols


def train_catboost_three_way(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_seed: int = 42,
    iterations: int = 5000,
    early_stopping_rounds: int = 200,
    verbose: int = 200,
    auto_class_weights: str = "Balanced",

    # Threshold policy (VAL only)
    decision_metric: str = "f1",  # "f1" or "cost"
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,

    # Regularization / stability defaults (used unless tuner overrides via model_kwargs)
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 8.0,
    min_data_in_leaf: int = 50,
    rsm: float = 0.9,
    random_strength: float = 2.0,
    model_size_reg: float = 0.5,

    **model_kwargs: Any,
) -> CatBoostThreeWayResult:
    """
    Production-correct protocol:
      - fit on TRAIN only (with VAL for early stopping)
      - choose threshold on VAL only
      - report once on TEST using locked VAL threshold
    """

    # Binary labels
    y_train_bin = (y_train == "Yes").astype(int).to_numpy()
    y_val_bin = (y_val == "Yes").astype(int).to_numpy()
    y_test_bin = (y_test == "Yes").astype(int).to_numpy()

    cat_cols = _infer_cat_features(X_train)

    # CatBoost wants indices for categorical columns (by position)
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    # Merge defaults with tuned params safely (tuned params win).
    extra: Dict[str, Any] = dict(model_kwargs)

    # Core hyperparams
    extra.setdefault("learning_rate", learning_rate)
    extra.setdefault("depth", depth)
    extra.setdefault("l2_leaf_reg", l2_leaf_reg)

    # Regularization / stability
    extra.setdefault("min_data_in_leaf", min_data_in_leaf)
    extra.setdefault("rsm", rsm)
    extra.setdefault("random_strength", random_strength)
    extra.setdefault("model_size_reg", model_size_reg)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(iterations),
        random_seed=int(random_seed),
        auto_class_weights=str(auto_class_weights),
        od_type="Iter",  # overfitting detector
        od_wait=int(early_stopping_rounds),
        verbose=int(verbose),
        allow_writing_files=False,
        thread_count=-1,
        **extra,
    )

    model.fit(
        X_train,
        y_train_bin,
        cat_features=cat_idx,
        eval_set=(X_val, y_val_bin),
        use_best_model=True,
    )

    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Threshold selection on VAL (policy controlled by decision_metric)
    rows = sweep_thresholds(y_val_bin, y_val_proba)

    metric = str(decision_metric).lower().strip()
    if metric == "cost":
        best_val = best_by_cost(rows, cost_fp=float(cost_fp), cost_fn=float(cost_fn))
    elif metric == "f1":
        best_val = best_by_f1(rows)
    else:
        raise ValueError(f"Unsupported decision_metric: {decision_metric}. Use 'f1' or 'cost'.")

    locked_t = float(best_val["threshold"])

    # Final TEST metrics at locked threshold
    test_pred = (y_test_proba >= locked_t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_bin, test_pred, labels=[0, 1]).ravel()

    test_metrics = {
        "threshold_locked_from_val": locked_t,
        "decision_metric": metric,
        "accuracy": float(accuracy_score(y_test_bin, test_pred)),
        "precision": float(precision_score(y_test_bin, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test_bin, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test_bin, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_bin, y_test_proba)),
        "pr_auc": float(average_precision_score(y_test_bin, y_test_proba)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    return CatBoostThreeWayResult(
        model=model,
        best_threshold_val=best_val,
        metrics_test_locked=test_metrics,
        y_val_proba=y_val_proba,
        y_test_proba=y_test_proba,
        cat_feature_names=cat_cols,
    )