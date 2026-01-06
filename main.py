from pathlib import Path

from src.config import load_config
from src.logger import get_logger
from src.data.load_data import load_raw_data
from src.data.clean import coerce_numeric_objects, drop_columns
from src.data.split_three_way import split_train_val_test
from src.models.catboost_tune_simple import tune_catboost_random
from src.models.catboost_three_way import train_catboost_three_way
from src.utils.io import save_metrics


def main() -> None:
    cfg = load_config()

    log_dir = (
        f"{cfg.paths.artifacts_dir}/logs" if hasattr(cfg.paths, "artifacts_dir") else "artifacts/logs"
    )
    logger = get_logger("customer_churn", log_dir=log_dir)

    # Handle schema naming (schema_ avoids pydantic BaseModel.schema clash)
    target_col = cfg.schema_.target_column if hasattr(cfg, "schema_") else cfg.schema.target_column

    logger.info("Starting Pipeline")
    logger.info("Raw data: %s", cfg.paths.raw_data)
    logger.info("Target column: %s", target_col)

    # Load
    df = load_raw_data(cfg.paths.raw_data, target_col)

    # Drop configured columns (e.g., customerID)
    df, drop_info = drop_columns(df, cfg.preprocess.drop_columns)
    logger.info("Drop result: %s", drop_info)

    # Coerce numeric-like object columns (e.g., TotalCharges)
    df = coerce_numeric_objects(df)
    if df is None:
        raise RuntimeError("df became None. Check clean.py functions return df.")

    logger.info("Dtypes after coercion:\n%s", df.dtypes)
    logger.info("Loaded data shape: %s", df.shape)
    logger.info("Churn distribution:\n%s", df[target_col].value_counts())

    # 3-way split (train/val/test) — NO test leakage
    # If val_size is missing in config, default to 0.10
    val_size = getattr(cfg.split, "val_size", 0.10)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df,
        target_col=target_col,
        test_size=cfg.split.test_size,
        val_size=val_size,
        random_seed=cfg.project.random_seed,
        stratify=cfg.split.stratify,
    )

    logger.info(
        "Train shape: %s | Val shape: %s | Test shape: %s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )
    logger.info("Train churn rate: %.4f", (y_train == "Yes").mean())
    logger.info("Val churn rate: %.4f", (y_val == "Yes").mean())
    logger.info("Test churn rate: %.4f", (y_test == "Yes").mean())
    logger.info("Stage 1 complete")

    # Artifacts dir
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # --- CatBoost: simple randomized tuning on VAL (NO test leakage) ---
    # Use cfg.tuning for trial settings, cfg.catboost for final-fit settings.
    cb_cfg = getattr(cfg, "catboost", None)
    cb_iterations = int(getattr(cb_cfg, "iterations", 20000))
    cb_early = int(getattr(cb_cfg, "early_stopping_rounds", 300))
    cb_verbose = int(getattr(cb_cfg, "verbose", 200))
    cb_auto_w = str(getattr(cb_cfg, "auto_class_weights", "Balanced"))

    tuning_cfg = getattr(cfg, "tuning", None)
    n_iter = int(getattr(tuning_cfg, "n_iter", 25))
    scorer = str(getattr(tuning_cfg, "scorer", "pr_auc"))
    tune_iterations = int(getattr(tuning_cfg, "iterations", 8000))
    tune_early = int(getattr(tuning_cfg, "early_stopping_rounds", 200))
    tune_verbose = int(getattr(tuning_cfg, "verbose", 0))

    tune = tune_catboost_random(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_seed=cfg.project.random_seed,
        n_iter=n_iter,
        scorer=scorer,
        iterations=tune_iterations,
        early_stopping_rounds=tune_early,
        verbose=tune_verbose,
        auto_class_weights=cb_auto_w,
    )

    logger.info("CatBoost tuning scorer: %s", scorer)
    logger.info("CatBoost tuning best params: %s", tune.best_params)

    # Note: tune.best_pr_auc_val stores the *selected* metric value (PR-AUC if scorer=pr_auc else ROC-AUC)
    best_selected = float(tune.best_pr_auc_val)
    logger.info(
        "Best VAL %s: %.6f | VAL ROC-AUC: %.6f",
        scorer,
        best_selected,
        float(tune.best_roc_auc_val),
    )

    save_metrics(
        {
            "scorer": scorer,
            "best_params": tune.best_params,
            "best_val_selected_score": best_selected,
            "best_roc_auc_val": float(tune.best_roc_auc_val),
            "trials": tune.trial_rows,
        },
        artifacts_dir / "metrics" / "catboost_random_tuning.json",
    )

    # --- Final CatBoost training + locked-threshold evaluation ---
    # train_catboost_three_way picks threshold on VAL (policy-controlled) and evaluates once on TEST.

    decision_cfg = getattr(cfg, "decision", None)
    decision_metric = str(getattr(decision_cfg, "metric", "f1"))
    cost_fp = float(getattr(decision_cfg, "cost_fp", 1.0))
    cost_fn = float(getattr(decision_cfg, "cost_fn", 5.0))

    result = train_catboost_three_way(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        random_seed=cfg.project.random_seed,
        iterations=cb_iterations,
        early_stopping_rounds=cb_early,
        verbose=cb_verbose,
        auto_class_weights=cb_auto_w,
        decision_metric=decision_metric,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        **tune.best_params,
    )

    logger.info("VAL best threshold (policy=%s): %s", decision_metric, result.best_threshold_val)
    logger.info("TEST metrics at locked VAL threshold: %s", result.metrics_test_locked)

    model_path = artifacts_dir / "models" / "catboost.cbm"
    result.model.save_model(str(model_path))

    save_metrics(
        result.best_threshold_val,
        artifacts_dir / "metrics" / "catboost_val_best_threshold.json",
    )
    save_metrics(
        result.metrics_test_locked,
        artifacts_dir / "metrics" / "catboost_test_metrics_locked_threshold.json",
    )
   


if __name__ == "__main__":
    main()
