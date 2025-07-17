import logging
import sys


def setup_logging():
    logger = logging.getLogger("copick_torch")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
