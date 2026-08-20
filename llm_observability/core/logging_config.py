"""
Logging setup for the whole project.

Why not just use print()? Because print() writes a line to the screen and
that's all it can ever do. A logger does more, for free:

  - it stamps every line with the date, the time, and how serious it is
  - it says which file the message came from
  - you can turn the noise up or down from ONE place, instead of hunting
    down and deleting print() calls scattered all over the project

Every file in this project starts with:

    logger = get_logger(__name__)

...and then uses logger.info(...) or logger.warning(...) instead of print().
"""

import logging

# Remembers whether we have already switched logging on. Without this, calling
# configure_logging() twice would set it up twice, and every message after
# that would be printed twice.
_CONFIGURED = False


def configure_logging(level=logging.INFO) -> None:
    """
    Switch logging on. Call this ONCE, right at the start of your program.

    Only the program should call this, never this library. How the log lines
    look is a decision for whoever is running the program, not for a library
    they installed. That is why main.py calls it and nothing inside the
    package does.

    `level` decides how much you see. logging.INFO shows normal progress
    messages and anything worse. logging.WARNING hides the normal ones and
    shows only problems.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        level=level,
        # This is the shape of each line. The %(...)s parts get filled in, so
        # a real line comes out looking like this:
        #
        #   2026-08-20 14:05:11 [INFO] llm_observability.api: Call succeeded
        #   |__ asctime _____|  |lvl|  |__ name ___________|  |_ message __|
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get the logger for one file.

    Always pass __name__ as the name. __name__ is a variable Python fills in
    for you with the current file's name, so your messages get labelled with
    where they came from and you never have to type it out.
    """
    return logging.getLogger(name)
