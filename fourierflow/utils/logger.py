
import logging

LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
}


def setup_logger(mode='info'):
    """Initialize logger. Mode can be: info, debug, warning, stackdriver."""
    logger = logging.getLogger()

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(LEVEL_DICT[mode])
    handler = logging.StreamHandler()

    # Format log messages
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
