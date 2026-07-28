"""
Central logging configuration.

Instead of print() scattered across the codebase, every module gets a
logger via get_logger(__name__), which:
- tags each line with a timestamp, severity level, and module name
- can be silenced or redirected in ONE place, without touching any
  module that logs something
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