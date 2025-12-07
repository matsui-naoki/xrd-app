"""
XRD Analyzer Logger Module
Provides logging functionality for the application
"""

import logging
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'xrd_app',
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_to_file: Whether to also log to a file
        log_file: Path to log file (default: xrd_app_YYYYMMDD.log)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        if log_file is None:
            log_file = f"xrd_app_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create default logger instance
logger = setup_logger()


def log_analysis_start(n_samples: int, n_components: int, distance_method: str):
    """Log analysis start"""
    logger.info(
        f"Starting analysis: {n_samples} samples, "
        f"{n_components} components, {distance_method} distance"
    )


def log_analysis_complete(n_clusters: int, error: float):
    """Log analysis completion"""
    logger.info(
        f"Analysis complete: {n_clusters} clusters found, "
        f"reconstruction error: {error:.2f}%"
    )


def log_preprocessing_start(n_files: int):
    """Log preprocessing start"""
    logger.info(f"Starting preprocessing for {n_files} files")


def log_preprocessing_complete():
    """Log preprocessing completion"""
    logger.info("Preprocessing complete")


def log_file_load(filename: str, n_points: int):
    """Log file loading"""
    logger.debug(f"Loaded {filename}: {n_points} data points")


def log_file_error(filename: str, error: str):
    """Log file loading error"""
    logger.warning(f"Failed to load {filename}: {error}")


def log_error(message: str, exc_info: bool = False):
    """Log an error"""
    logger.error(message, exc_info=exc_info)


def log_warning(message: str):
    """Log a warning"""
    logger.warning(message)


def log_info(message: str):
    """Log an info message"""
    logger.info(message)


def log_debug(message: str):
    """Log a debug message"""
    logger.debug(message)
