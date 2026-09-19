"""Structured logging configuration for the Airline Delay Prediction project."""

import logging
import os
from pathlib import Path
import sys
from typing import Optional

# Standard format string
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "airline_delay_prediction",
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configures and returns a thread-safe logger with console and file handlers.
    
    Args:
        name: Name of the logger (typically module name or project identifier).
        log_level: Desired log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to output log file.

    Returns:
        Configured logging.Logger instance.
    """
    level_str = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler if specified or defaulted
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # Do not propagate to root logger to avoid double logging
    logger.propagate = False
    return logger
