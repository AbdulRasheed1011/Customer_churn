from src.config import load_config
from src.logger import get_logger

def main():
    cfg = load_config()
    log_dir = f'{cfg.paths.artifacts_dir}/logs' if hasattr(cfg.paths, 'artifacts_dir') else 'artifacts/logs'
    logger = get_logger('customer_churn', log_dir = log_dir)

    logger.info('Starting Pipeline')
    logger.info('Raw data: %s', cfg.paths.raw_data)
    logger.info('Target column: %s', cfg.schema_.target_column if hasattr(cfg, 'schema_') else cfg.schema.target_column)


if __name__ == '__main__':
    main()
