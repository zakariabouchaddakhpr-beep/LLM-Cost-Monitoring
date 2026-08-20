import logging

_CONFIGURED = False


def configure_logging(level=logging.INFO) -> None:
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
