from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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


def _make_ohe() -> OneHotEncoder:
    """Create a dense OneHotEncoder across sklearn versions."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass
class TrainResult:
    model: Pipeline
    metrics: Dict[str, Any]
    y_test_true: np.ndarray
    y_test_proba: np.ndarray


def train_hgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_seed: int = 42,
) -> TrainResult:
    """Train HistGradientBoosting with preprocessing (impute + OHE)."""

    # Map labels to 0/1
    y_train_bin = (y_train == "Yes").astype(int).to_numpy()
    y_test_bin = (y_test == "Yes").astype(int).to_numpy()

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_ohe()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        random_state=random_seed,
    )

    model = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    model.fit(X_train, y_train_bin)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test_bin, pred)),
        "precision": float(precision_score(y_test_bin, pred, zero_division=0)),
        "recall": float(recall_score(y_test_bin, pred, zero_division=0)),
        "f1": float(f1_score(y_test_bin, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_bin, proba)),
        "pr_auc": float(average_precision_score(y_test_bin, proba)),
        "confusion_matrix": confusion_matrix(y_test_bin, pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "num_features": int(len(num_cols)),
        "cat_features": int(len(cat_cols)),
    }

    return TrainResult(
        model=model,
        metrics=metrics,
        y_test_true=y_test_bin,
        y_test_proba=proba,
    )