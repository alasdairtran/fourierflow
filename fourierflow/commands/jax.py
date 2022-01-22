import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, cast
from uuid import uuid4

import hydra
import ptvsd
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typer import Argument, Typer

from fourierflow.utils import delete_old_results

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
        wandb_opts = cast(dict, OmegaConf.to_container(config.wandb))
        wandb.init(dir=config_dir,
                   mode=os.environ.get('WANDB_MODE', 'offline'),
                   config=deepcopy(cast(dict, OmegaConf.to_container(config))),
                   id=str(uuid4())[:8],
                   **wandb_opts)

    # Initialize the dataset and experiment modules.
    builder = instantiate(config.builder)
    routine = instantiate(config.routine)

    routine.fit(
        inputs=builder.train_dataloader(),
        epochs=2,
        steps_per_epoch=200,
        batch_size=1,
        validation_data=builder.val_dataloader(),
        shuffle=True,
        # callbacks=[eg.callbacks.TensorBoard("summaries")]
    )


if __name__ == "__main__":
    app()
