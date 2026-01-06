# src/models/catboost_tune_simple.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class CatBoostTuneResult:
    best_params: Dict[str, Any]
    best_pr_auc_val: float
    best_roc_auc_val: float
    trial_rows: List[Dict[str, Any]]


def infer_cat_features(X: pd.DataFrame) -> Tuple[List[str], List[int]]:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    return cat_cols, cat_idx


def _sample_params(rng: np.random.Generator) -> Dict[str, Any]:

    depth = int(rng.choice([4, 5, 6, 7, 8, 9, 10]))
    learning_rate = float(10 ** rng.uniform(-2.0, -0.7))  # ~0.01 to 0.2 (log-uniform)
    l2_leaf_reg = float(10 ** rng.uniform(0.0, 1.3))      # ~1 to 20 (log-uniform)
    min_data_in_leaf = int(rng.choice([20, 30, 50, 80, 100, 150, 200]))
    rsm = float(rng.uniform(0.7, 1.0))

    bootstrap_type = str(rng.choice(["Bayesian", "Bernoulli"]))
    params: Dict[str, Any] = {
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg,
        "min_data_in_leaf": min_data_in_leaf,
        "rsm": rsm,
        "bootstrap_type": bootstrap_type,
    }

    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = float(rng.uniform(0.0, 10.0))
    else:
        params["subsample"] = float(rng.uniform(0.6, 0.9))

    return params


def tune_catboost_random(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    random_seed: int = 42,
    n_iter: int = 25,
    scorer: str = "pr_auc",  # "pr_auc" or "roc_auc" (controls which metric selects best params)
    iterations: int = 5000,
    early_stopping_rounds: int = 200,
    verbose: int = 0,
    auto_class_weights: str = "Balanced",
) -> CatBoostTuneResult:
    """
    Random search:
      - train on TRAIN
      - evaluate PR-AUC on VAL
      - pick best hyperparams by PR-AUC (Average Precision)
    """
    rng = np.random.default_rng(random_seed)

    y_train_bin = (y_train == "Yes").astype(int).to_numpy()
    y_val_bin = (y_val == "Yes").astype(int).to_numpy()

    _, cat_idx = infer_cat_features(X_train)

    best_params: Optional[Dict[str, Any]] = None
    best_pr = -1.0
    best_roc = -1.0
    rows: List[Dict[str, Any]] = []

    scorer = scorer.lower().strip()
    if scorer not in {"pr_auc", "roc_auc"}:
        raise ValueError(f"Unsupported scorer: {scorer}. Use 'pr_auc' or 'roc_auc'.")

    for i in range(1, n_iter + 1):
        sampled = _sample_params(rng)

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=iterations,
            random_seed=random_seed,
            auto_class_weights=auto_class_weights,
            verbose=verbose,
            allow_writing_files=False,
            thread_count=-1,
            od_type="Iter",
            od_wait=early_stopping_rounds,
            **sampled,
        )

        model.fit(
            X_train,
            y_train_bin,
            cat_features=cat_idx,
            eval_set=(X_val, y_val_bin),
            use_best_model=True,
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        pr = float(average_precision_score(y_val_bin, val_proba))
        roc = float(roc_auc_score(y_val_bin, val_proba))

        # Best iteration can be None depending on settings/version; keep it robust.
        try:
            best_it = model.get_best_iteration()
            best_it = int(best_it) if best_it is not None else -1
        except Exception:
            best_it = -1

        row = {
            "trial": i,
            "pr_auc_val": pr,
            "roc_auc_val": roc,
            "score_selected": pr if scorer == "pr_auc" else roc,
            "best_iteration": best_it,
            **sampled,
        }
        rows.append(row)

        score_selected = pr if scorer == "pr_auc" else roc
        if score_selected > best_pr:
            # Keep best_pr as 'best selected score' (name kept for backward compatibility)
            best_pr = score_selected
            best_roc = roc
            best_params = sampled

    assert best_params is not None
    return CatBoostTuneResult(
        best_params=best_params,
        best_pr_auc_val=best_pr,  # selected metric value (PR-AUC if scorer=pr_auc else ROC-AUC)
        best_roc_auc_val=best_roc,
        trial_rows=rows,
    )