"""
utils/logger.py — Loguru setup for the entire project.
Import get_logger() in any module instead of using logging directly.
"""

import sys
from pathlib import Path
from loguru import logger

from config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT, LOG_ROTATION, LOG_RETENTION


def setup_logger(name: str = "btc_predictor") -> None:
    """
    Call once from main.py to configure all sinks.
    Subsequent get_logger() calls just return the shared logger.
    """
    logger.remove()  # remove default stderr sink

    # Console sink — coloured, human-readable
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        colorize=True,
    )

    # File sink — rotates daily, keeps 30 days
    log_file = LOGS_DIR / f"{name}.log"
    logger.add(
        log_file,
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        compression="zip",
        enqueue=True,       # thread-safe async writes
    )


def get_logger():
    """Return the shared loguru logger instance."""
    return logger