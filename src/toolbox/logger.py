import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "Classifier_Model",
    log_level: str = "DEBUG",
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Args:
        name: Logger name
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (if None, console only)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # set logger level
    effective_level = getattr(logging, log_level.upper())
    logger.setLevel(effective_level)

    # turn off displaying same log by different loggers (for colab)
    logger.propagate = False

    if log_dir is None and logger.hasHandlers():
        return logger

    # delete old hanlders (for notebooks)
    if logger.hasHandlers():
        logger.handlers.clear()

    # console handler (always active)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(effective_level)
    console_format = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # file handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # log format
        file_format = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        main_handler = logging.handlers.RotatingFileHandler(
            log_path / "main.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
            encoding="utf-8",
        )
        main_handler.setLevel(effective_level)
        main_handler.setFormatter(file_format)
        logger.addHandler(main_handler)

        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "error.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        logger.addHandler(error_handler)

    return logger