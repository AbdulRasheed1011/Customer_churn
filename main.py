from pathlib import Path

from src.config import load_config
from src.logger import get_logger
from src.data.load_data import load_raw_data
from src.data.clean import coerce_numeric_objects, drop_columns
from src.data.split_three_way import split_train_val_test
from src.models.hgb_three_way import train_hgb_three_way
from src.utils.io import save_model, save_metrics


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

    # Train HGB (fit on train, pick threshold on val, evaluate once on test)
    # If you later add cfg.hgb, wire params here.
    result = train_hgb_three_way(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        random_seed=cfg.project.random_seed,
    )

    logger.info("VAL best threshold (F1): %s", result.best_threshold_val)
    logger.info("TEST metrics at locked VAL threshold: %s", result.metrics_test_locked)

    artifacts_dir = Path(cfg.paths.artifacts_dir)
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)

    save_model(result.model, artifacts_dir / "models" / "hgb.joblib")
    save_metrics(result.metrics_train_val, artifacts_dir / "metrics" / "hgb_val_metrics.json")
    save_metrics(result.best_threshold_val, artifacts_dir / "metrics" / "hgb_val_best_threshold.json")
    save_metrics(result.metrics_test_locked, artifacts_dir / "metrics" / "hgb_test_metrics_locked_threshold.json")

    logger.info("Saved HGB artifacts under: %s", artifacts_dir.resolve())


if __name__ == "__main__":
    main()
