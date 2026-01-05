import inspect
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config
from src.data.clean import coerce_numeric_objects, drop_columns
from src.data.load_data import load_raw_data
from src.data.split import split_train_test
from src.eval.threshold import best_by_f1, sweep_thresholds
from src.logger import get_logger
from src.models.hgb import train_hgb
from src.utils.io import save_metrics, save_model


def _call_train_hgb(
    X_train,
    y_train,
    X_test,
    y_test,
    seed: int,
):

    try:
        sig = inspect.signature(train_hgb)
    except Exception:
        sig = None

    # Prefer keyword arguments when possible (more robust than positional).
    kwargs = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

    if sig and "random_seed" in sig.parameters:
        kwargs["random_seed"] = seed
    elif sig and "random_state" in sig.parameters:
        kwargs["random_state"] = seed

    try:
        return train_hgb(**kwargs)
    except TypeError:
        # Fallback to pure positional call if the local signature is positional-only.
        return train_hgb(X_train, y_train, X_test, y_test)


def main() -> None:
    cfg = load_config()

    log_dir = (
        f"{cfg.paths.artifacts_dir}/logs" if hasattr(cfg.paths, "artifacts_dir") else "artifacts/logs"
    )
    logger = get_logger("customer_churn", log_dir=log_dir)

    logger.info("Starting Pipeline")
    logger.info("Raw data: %s", cfg.paths.raw_data)

    # schema may be aliased as schema_ to avoid pydantic BaseModel.schema clash
    target_col = cfg.schema_.target_column if hasattr(cfg, "schema_") else cfg.schema.target_column
    logger.info("Target column: %s", target_col)

    # Load data
    df = load_raw_data(cfg.paths.raw_data, target_col)

    # Drop ID/leakage columns
    df, drop_info = drop_columns(df, cfg.preprocess.drop_columns)
    logger.info("Drop result: %s", drop_info)

    # Coerce numeric-like object columns (e.g., TotalCharges)
    df = coerce_numeric_objects(df)
    if df is None:
        raise RuntimeError("df became None. A preprocessing function is missing `return df`.")

    logger.info("Dtypes after coercion:\n%s", df.dtypes)
    logger.info("Loaded data shape: %s", df.shape)
    logger.info("Churn distribution:\n%s", df[target_col].value_counts())

    # Split
    X_train, X_test, y_train, y_test = split_train_test(
        df=df,
        target_col=target_col,
        test_size=cfg.split.test_size,
        random_seed=cfg.project.random_seed,
        stratify=cfg.split.stratify,
    )

    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)
    logger.info("Train churn rate: %.4f", (y_train == "Yes").mean())
    logger.info("Test churn rate: %.4f", (y_test == "Yes").mean())
    logger.info("Stage 1 complete")

    # ---- Model: HistGradientBoosting ----
    out = _call_train_hgb(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=cfg.project.random_seed,
    )

    if hasattr(out, "y_test_proba") and hasattr(out, "model"):
        model = out.model
        proba = out.y_test_proba
        model_metrics = getattr(out, "metrics", {})
        logger.info("HGB metrics: %s", model_metrics)
    else:
        model = out
        proba = model.predict_proba(X_test)[:, 1]
        model_metrics = {}

    y_test_bin = (y_test == "Yes").astype(int).to_numpy()

    holdout_metrics = {
        "pr_auc_test": float(average_precision_score(y_test_bin, proba)),
        "roc_auc_test": float(roc_auc_score(y_test_bin, proba)),
    }
    logger.info("HGB holdout metrics: %s", holdout_metrics)

    rows = sweep_thresholds(y_test_bin, proba)
    best = best_by_f1(rows)
    logger.info("Best threshold by F1 (HGB): %s", best)

    # Save artifacts
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    save_model(model, artifacts_dir / "models" / "hgb.joblib")

    # Save any model-provided metrics plus holdout/threshold metrics
    if model_metrics:
        save_metrics(model_metrics, artifacts_dir / "metrics" / "hgb_metrics.json")
    save_metrics(holdout_metrics, artifacts_dir / "metrics" / "hgb_holdout_metrics.json")
    save_metrics({"best": best, "sweep": rows}, artifacts_dir / "metrics" / "hgb_threshold_sweep.json")

    logger.info("Saved HGB artifacts under: %s", artifacts_dir.resolve())


if __name__ == "__main__":
    main()
