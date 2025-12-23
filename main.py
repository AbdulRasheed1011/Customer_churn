from src.config import load_config
from src.logger import get_logger
from src.data.load_data import load_raw_data
from src.data.split import split_train_test

def main() -> None:
    cfg = load_config()
    log_dir = f'{cfg.paths.artifacts_dir}/logs' if hasattr(cfg.paths, 'artifacts_dir') else 'artifacts/logs'
    logger = get_logger('customer_churn', log_dir = log_dir)

    logger.info('Starting Pipeline')
    logger.info('Raw data: %s', cfg.paths.raw_data)
    logger.info('Target column: %s', cfg.schema_.target_column if hasattr(cfg, 'schema_') else cfg.schema.target_column)

    df = load_raw_data(cfg.paths.raw_data, cfg.schema_.target_column)
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
if __name__ == '__main__':
    main()
