import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, cast

import hydra
import jax
import numpy as np
import ptvsd
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from typer import Argument, Typer

from fourierflow.trainers import JAXTrainer
from fourierflow.utils import (delete_old_results, get_experiment_id,
                               upload_code_to_wandb)

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: Path,
         overrides: Optional[List[str]] = Argument(None),
         force: bool = False,
         trial: int = 0,
         debug: bool = False,
         no_logging: bool = False):
    """Train a JAX experiment."""
    config_dir = config_path.parent
    config_name = config_path.stem
    hydra.initialize(config_path=str(Path('../..') / config_dir))
    config = hydra.compose(config_name, overrides=overrides or [])
    OmegaConf.set_struct(config, False)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()
        jax.config.update('jax_disable_jit', True)

    # Set up directories to save experimental outputs.
    delete_old_results(config_dir, force, trial, resume=False)

    # Set seed for reproducibility.
    rs = np.random.RandomState(7231 + trial)
    seed = config.get('seed', rs.randint(1000, 1000000))
    pl.seed_everything(seed, workers=True)
    config.seed = seed
    wandb_id = get_experiment_id(None, trial, config_dir, False)
    config.trial = trial

    if no_logging:
        logger = None
    else:
        config.wandb.name = f"{config.wandb.group}/{trial}"
        wandb_opts = cast(dict, OmegaConf.to_container(config.wandb))
        logger = WandbLogger(save_dir=str(config_dir),
                             mode=os.environ.get('WANDB_MODE', 'offline'),
                             config=deepcopy(OmegaConf.to_container(config)),
                             id=wandb_id,
                             **wandb_opts)
        upload_code_to_wandb(Path(config_dir) / 'config.yaml', logger)
        c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
        c.cleanup(wandb.util.from_human_size("100GB"))

    # Initialize the dataset and experiment modules.
    builder = instantiate(config.builder)
    routine = instantiate(config.routine)
    callbacks = [instantiate(p) for p in config.trainer.pop('callbacks', [])]

    trainer = JAXTrainer(callbacks=callbacks,
                         seed=seed,
                         logger=logger,
                         trial=trial,
                         **OmegaConf.to_container(config.trainer))
    trainer.fit(routine, builder)
    trainer.test(routine, builder)


if __name__ == "__main__":
    app()
