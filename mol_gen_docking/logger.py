"""Logging utilities for the MolGenDocking package."""

import logging


def create_logger(name: str, level: str = "INFO"):
    """
    Create a logger object with the specified name and level.
    :param name: Name of the logger
    :param level: Level of the logger
    :return: Logger object
    """
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s | %(name)s | %(levelname)s] %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(level)
    return logger
