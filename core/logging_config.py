"""
logger via get_logger(__name__)
"""

import logging

_CONFIGURED = False


def configure_logging(level=logging.INFO) -> None:
    """Call this once, at the entry point of the application
    (e.g. quickstart.py), before anything else logs."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)