from src.config import load_config
from src.logger import get_logger
from src.data.load_data import load_raw_data
from src.data.split import split_train_test
from pathlib import Path
from src.data.clean import coerce_numeric_objects, drop_columns
from src.models.baseline import train_baseline
from src.utils.io import save_model, save_metrics
from src.eval.threshold import sweep_thresholds, best_by_f1
from src.models.tune import tune_logreg_pr_auc
from sklearn.metrics import average_precision_score, roc_auc_score

def main() -> None:
    cfg = load_config()
    log_dir = f'{cfg.paths.artifacts_dir}/logs' if hasattr(cfg.paths, 'artifacts_dir') else 'artifacts/logs'
    logger = get_logger('customer_churn', log_dir = log_dir)

    logger.info('Starting Pipeline')
    logger.info('Raw data: %s', cfg.paths.raw_data)
    logger.info('Target column: %s', cfg.schema_.target_column if hasattr(cfg, 'schema_') else cfg.schema.target_column)

    df = load_raw_data(cfg.paths.raw_data, cfg.schema_.target_column)

    ## column drop

    df, drop_info = drop_columns(df, cfg.preprocess.drop_columns)

    df = coerce_numeric_objects(df)
    
    if df is None:
        raise RuntimeError("df became None. Check clean.py functions return df.")

    logger.info('Dtypes after coercion: \n%s', df.dtypes)

    logger.info('Loaded data shape: %s', df.shape)
    logger.info('Churn distribution : \n%s', df[cfg.schema_.target_column].value_counts())

    X_train, X_test, y_train, y_test = split_train_test(
        df = df,
        target_col = cfg.schema_.target_column,
        test_size = cfg.split.test_size,
        random_seed = cfg.project.random_seed,
        stratify = cfg.split.stratify,
    )

    logger.info('Train shape : %s | Test shape: %s', X_train.shape, X_test.shape)
    logger.info('Train Churn rate : %.4f', (y_train == 'Yes').mean())
    logger.info('Test churn rate : %.4f', (y_test == 'Yes').mean())
    logger.info('Stage 1 complete')



    best_model, cv_report = tune_logreg_pr_auc(
        X_train = X_train,
        y_train = y_train,
        random_seed = cfg.project.random_seed,
        n_iter = 30,
    )

    logger.info('Tuning report : %s', cv_report)

    y_test_bin = (y_test == 'Yes').astype(int).to_numpy()
    proba = best_model.predict_proba(X_test)[:,1]

    test_metrics = {
        'pr_auc_test' : float(average_precision_score(y_test_bin, proba)),
        'roc_auc_test' : float(roc_auc_score(y_test_bin, proba)),
    }

    logger.info('Tuned holdout metrics: %s', test_metrics)
    rows = sweep_thresholds(y_test_bin, proba)
    best = best_by_f1(rows)
    logger.info('Best threshold by F1 (tuned model): %s', best)
if __name__ == '__main__':
    main()
