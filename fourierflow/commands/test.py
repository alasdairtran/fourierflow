import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, cast

import hydra
import numpy as np
import ptvsd
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from typer import Argument, Typer

from fourierflow.utils import (delete_old_results, get_experiment_id,
                               import_string)

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: Path,
         overrides: Optional[List[str]] = Argument(None),
         force: bool = False,
         trial: int = 0,
         map_location: Optional[str] = None,
         remove_keys: Optional[str] = None,
         debug: bool = False,
         no_logging: bool = False):
    """Test a Pytorch Lightning experiment."""
    config_dir = config_path.parent
    config_name = config_path.stem
    hydra.initialize(config_path=Path('../..') / config_dir)
    config = hydra.compose(config_name, overrides=overrides)
    OmegaConf.set_struct(config, False)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()
        # ptvsd doesn't play well with multiple processes.
        config.builder.num_workers = 0

    # We use Weights & Biases to track our experiments.
    checkpoint_path = config.get('checkpoint_path', None)
    if not checkpoint_path:
        chkpt_dir = Path(config_dir) / 'checkpoints'
        paths = list(chkpt_dir.glob(f'trial-{trial}-*/epoch*.ckpt'))
        # if len(paths) > 1:
        #     paths = list(chkpt_dir.glob(f'trial-{trial}-*/last.ckpt'))
        assert len(paths) == 1
        checkpoint_path = paths[0]
        wandb_id = Path(checkpoint_path).parent.name
        trial = int(wandb_id.split('-')[1])
    else:
        delete_old_results(config_dir, force, trial, resume=False)
        wandb_id = get_experiment_id(None, trial, config_dir, resume=False)
    config.trial = trial
    config.wandb.name = f"{config.wandb.group}/{trial}"
    wandb_opts = cast(dict, OmegaConf.to_container(config.wandb))
    wandb_logger = WandbLogger(save_dir=str(config_dir),
                               mode=os.environ.get('WANDB_MODE', 'offline'),
                               config=deepcopy(OmegaConf.to_container(config)),
                               id=wandb_id,
                               **wandb_opts)

    # Set seed for reproducibility.
    rs = np.random.RandomState(7231 + trial)
    seed = config.get('seed', rs.randint(1000, 1000000))
    pl.seed_everything(seed, workers=True)
    config.seed = seed
    if 'seed' in config.trainer:
        config.trainer.seed = seed

    builder = instantiate(config.builder)
    routine = instantiate(config.routine)
    remove_keys = remove_keys.split(',') if remove_keys else []
    routine.load_lightning_model_state(
        str(checkpoint_path), map_location, remove_keys=remove_keys)

    # Start the main testing pipeline.
    Trainer = import_string(config.trainer.pop(
        '_target_', 'pytorch_lightning.Trainer'))

    if no_logging:
        trainer = Trainer(logger=False, enable_checkpointing=False,
                          **OmegaConf.to_container(config.trainer))
    else:
        trainer = Trainer(logger=wandb_logger,
                          **OmegaConf.to_container(config.trainer))
    trainer.test(routine, datamodule=builder)


if __name__ == "__main__":
    app()
