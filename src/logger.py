import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(
    name: str,
    log_dir: str = "artifacts/logs",
    level: int = logging.INFO,
    filename: str = "run.log",
) -> logging.Logger:

    logger = logging.getLogger(name)

    # If handlers already exist, logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler (rotates)
    fh = RotatingFileHandler(
        log_path / filename,
        maxBytes=2_000_000,  # ~2MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Prevent double logging via root logger
    logger.propagate = False
    return logger