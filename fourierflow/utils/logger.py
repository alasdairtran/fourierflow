
import logging
from glob import glob

import wandb

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


def upload_code_to_wandb(config_path, wandb_logger):
    """Upload all Python code for save the exact state of the experiment."""
    code_artifact = wandb.Artifact('fourierflow', type='code')
    code_artifact.add_file(config_path, 'config.yaml')
    paths = glob('fourierflow/**/*.py')
    for path in paths:
        code_artifact.add_file(path, path)
    wandb_logger.experiment.log_artifact(code_artifact)
