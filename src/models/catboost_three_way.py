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


def _best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
    thresholds = np.linspace(0.0, 1.0, 101)
    best = None

    for t in thresholds:
        pred = (proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if best is None or f1 > best["f1"]:
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            best = {
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }

    assert best is not None
    return best


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
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    early_stopping_rounds: int = 200,
    verbose: int = 200,
    auto_class_weights: str = 'Balanced',
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

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        auto_class_weights=auto_class_weights,     # handles imbalance sensibly
        od_type="Iter",                    # overfitting detector
        od_wait=early_stopping_rounds,
        verbose=verbose,
        **model_kwargs,
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

    # Threshold selection on VAL
    best_val = _best_threshold_by_f1(y_val_bin, y_val_proba)
    locked_t = float(best_val["threshold"])

    # Final TEST metrics at locked threshold
    test_pred = (y_test_proba >= locked_t).astype(int)
    test_metrics = {
        "threshold_locked_from_val": locked_t,
        "accuracy": float(accuracy_score(y_test_bin, test_pred)),
        "precision": float(precision_score(y_test_bin, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test_bin, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test_bin, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_bin, y_test_proba)),
        "pr_auc": float(average_precision_score(y_test_bin, y_test_proba)),
        "confusion_matrix": confusion_matrix(y_test_bin, test_pred).tolist(),
    }

    return CatBoostThreeWayResult(
        model=model,
        best_threshold_val=best_val,
        metrics_test_locked=test_metrics,
        y_val_proba=y_val_proba,
        y_test_proba=y_test_proba,
        cat_feature_names=cat_cols,
    )