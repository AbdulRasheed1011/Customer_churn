from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )
    return preprocessor, num_cols, cat_cols


def tune_logreg_pr_auc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int = 42,
    n_iter: int = 30,
) -> Tuple[Pipeline, Dict[str, Any]]:
    y_train_bin = (y_train == "Yes").astype(int).to_numpy()

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=4000, solver="saga", class_weight="balanced")),
        ]
    )

    # Search space as a list of dicts:
    # - l1_ratio sampled ONLY when penalty='elasticnet'
    param_dist = [
        {   # l1 / l2: NO l1_ratio
            "clf__penalty": ["l1", "l2"],
            "clf__C": np.logspace(-3, 2, 30),
        },
        {   # elasticnet: YES l1_ratio
            "clf__penalty": ["elasticnet"],
            "clf__C": np.logspace(-3, 2, 30),
            "clf__l1_ratio": np.linspace(0.0, 1.0, 6),
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",  # PR-AUC
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=random_seed,
        refit=True,
    )

    search.fit(X_train, y_train_bin)

    report = {
        "best_score_pr_auc_cv": float(search.best_score_),
        "best_params": search.best_params_,
        "num_features": int(len(num_cols)),
        "cat_features": int(len(cat_cols)),
    }

    return search.best_estimator_, report