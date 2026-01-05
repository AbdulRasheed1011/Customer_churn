from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class HGBThreeWayResult:
    model: Pipeline
    metrics_train_val: Dict[str, Any]          # metrics at default threshold on VAL (optional but useful)
    best_threshold_val: Dict[str, Any]         # threshold chosen on VAL (e.g., by F1)
    metrics_test_locked: Dict[str, Any]        # TEST metrics evaluated at the locked VAL threshold
    y_val_proba: np.ndarray
    y_test_proba: np.ndarray


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # HGB needs numeric input: we must one-hot encode categoricals
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols


def train_hgb_three_way(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_seed: int = 42,
    max_iter: int = 300,
    learning_rate: float = 0.1,
) -> HGBThreeWayResult:
    """
    PRODUCTION-CORRECT protocol:
      - fit on TRAIN only
      - choose threshold on VAL only
      - report final metrics once on TEST at locked threshold
    """
    # binarize target
    y_train_bin = (y_train == "Yes").astype(int).to_numpy()
    y_val_bin = (y_val == "Yes").astype(int).to_numpy()
    y_test_bin = (y_test == "Yes").astype(int).to_numpy()

    preprocessor, num_cols, cat_cols = _build_preprocessor(X_train)

    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        random_state=random_seed,
    )

    model = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    model.fit(X_train, y_train_bin)

    # Probabilities
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # --- threshold selection on VAL ---
    # keep this simple: sweep thresholds 0.00..1.00 step 0.01
    thresholds = np.linspace(0.0, 1.0, 101)
    best = None

    for t in thresholds:
        pred = (y_val_proba >= t).astype(int)
        p = precision_score(y_val_bin, pred, zero_division=0)
        r = recall_score(y_val_bin, pred, zero_division=0)
        f1 = f1_score(y_val_bin, pred, zero_division=0)

        if best is None or f1 > best["f1"]:
            tn, fp, fn, tp = confusion_matrix(y_val_bin, pred).ravel()
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
    locked_t = float(best["threshold"])

    # Optional: VAL metrics at default 0.5 (useful for debugging)
    val_pred_05 = (y_val_proba >= 0.5).astype(int)
    metrics_train_val = {
        "val_threshold_default": 0.5,
        "val_accuracy": float(accuracy_score(y_val_bin, val_pred_05)),
        "val_precision": float(precision_score(y_val_bin, val_pred_05, zero_division=0)),
        "val_recall": float(recall_score(y_val_bin, val_pred_05, zero_division=0)),
        "val_f1": float(f1_score(y_val_bin, val_pred_05, zero_division=0)),
        "val_roc_auc": float(roc_auc_score(y_val_bin, y_val_proba)),
        "val_pr_auc": float(average_precision_score(y_val_bin, y_val_proba)),
        "num_features": int(len(num_cols)),
        "cat_features": int(len(cat_cols)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }

    # --- final TEST metrics at locked threshold (NO selection here) ---
    test_pred = (y_test_proba >= locked_t).astype(int)
    metrics_test_locked = {
        "threshold_locked_from_val": locked_t,
        "accuracy": float(accuracy_score(y_test_bin, test_pred)),
        "precision": float(precision_score(y_test_bin, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test_bin, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test_bin, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_bin, y_test_proba)),
        "pr_auc": float(average_precision_score(y_test_bin, y_test_proba)),
        "confusion_matrix": confusion_matrix(y_test_bin, test_pred).tolist(),
    }

    return HGBThreeWayResult(
        model=model,
        metrics_train_val=metrics_train_val,
        best_threshold_val=best,
        metrics_test_locked=metrics_test_locked,
        y_val_proba=y_val_proba,
        y_test_proba=y_test_proba,
    )