import logging
import os
import sys

def setup_logger(save_dir, name="train"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent double logging if parent logger is configured

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, f"{name}.log")
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
