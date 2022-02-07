import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, cast
from uuid import uuid4

import hydra
import jax
import ptvsd
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typer import Argument, Typer

from fourierflow.trainers import JAXTrainer
from fourierflow.utils import delete_old_results, get_experiment_id

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

    if not no_logging:
        wandb_id = get_experiment_id(None, trial, config_dir, False)
        config.trial = trial
        config.wandb.name = f"{config.wandb.group}/{trial}"
        wandb_opts = cast(dict, OmegaConf.to_container(config.wandb))
        wandb.init(dir=config_dir,
                   mode=os.environ.get('WANDB_MODE', 'offline'),
                   config=deepcopy(cast(dict, OmegaConf.to_container(config))),
                   id=wandb_id,
                   **wandb_opts)

    # Initialize the dataset and experiment modules.
    builder = instantiate(config.builder)
    routine = instantiate(config.routine)
    callbacks = [instantiate(p) for p in config.trainer.pop('callbacks', [])]

    trainer = JAXTrainer(callbacks=callbacks,
                         **OmegaConf.to_container(config.trainer))
    trainer.fit(routine, builder)

    # routine.fit(
    #     inputs=builder.train_dataloader(),
    #     validation_data=builder.val_dataloader(),
    #     callbacks=callbacks,
    #     **OmegaConf.to_container(config.trainer),
    # )

    # logs = routine.evaluate(
    #     x=builder.test_dataloader(),
    #     callbacks=callbacks,
    #     drop_remaining=False,
    # )

    # logs = {"test_" + name: val for name, val in logs.items()}
    # print(logs)


if __name__ == "__main__":
    app()
