import os
import shutil
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import List, Optional, cast

import hydra
import numpy as np
import ptvsd
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from typer import Argument, Typer

from fourierflow.utils import ExistingExperimentFound, get_experiment_id

app = Typer()


def delete_old_results(results_dir, force, trial, resume):
    """Delete existing checkpoints and wandb logs if --force is enabled."""
    wandb_dir = Path(results_dir) / 'wandb'
    wandb_matches = list(wandb_dir.glob(f'*-trial-{trial}-*'))

    chkpt_dir = Path(results_dir) / 'checkpoints'
    chkpt_matches = list(chkpt_dir.glob(f'trial-{trial}-*'))

    if force and wandb_matches:
        [shutil.rmtree(p) for p in wandb_matches]

    if force and chkpt_matches:
        [shutil.rmtree(p) for p in chkpt_matches]

    if not force and not resume and wandb_matches:
        raise ExistingExperimentFound(f'Directory already exists: {wandb_dir}')

    if not force and not resume and chkpt_matches:
        raise ExistingExperimentFound(f'Directory already exists: {chkpt_dir}')


def upload_code_to_wandb(config_path, wandb_logger):
    """Upload all Python code for save the exact state of the experiment."""
    code_artifact = wandb.Artifact('fourierflow', type='code')
    code_artifact.add_file(config_path, 'config.yaml')
    paths = glob('fourierflow/**/*.py')
    for path in paths:
        code_artifact.add_file(path, path)
    wandb_logger.experiment.log_artifact(code_artifact)


@app.callback(invoke_without_command=True)
def main(config_path: Path,
         overrides: Optional[List[str]] = Argument(None),
         force: bool = False,
         resume: bool = False,
         checkpoint_id: Optional[str] = None,
         trial: int = 0,
         debug: bool = False):
    """Train a Pytorch Lightning experiment."""
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

    # Set up directories to save experimental outputs.
    delete_old_results(config_dir, force, trial, resume)

    # Set seed for reproducibility.
    rs = np.random.RandomState(7231 + trial)
    seed = config.get('seed', rs.randint(1000, 1000000))
    pl.seed_everything(seed, workers=True)
    config.seed = seed

    # We use Weights & Biases to track our experiments.
    wandb_id = get_experiment_id(checkpoint_id, trial, config_dir, resume)
    config.trial = trial
    config.wandb.name = f"{config.wandb.group}/{trial}"
    wandb_opts = cast(dict, OmegaConf.to_container(config.wandb))
    wandb_logger = WandbLogger(save_dir=str(config_dir),
                               mode=os.environ.get('WANDB_MODE', 'offline'),
                               config=deepcopy(OmegaConf.to_container(config)),
                               id=wandb_id,
                               **wandb_opts)
    upload_code_to_wandb(Path(config_dir) / 'config.yaml', wandb_logger)

    # Initialize the dataset and experiment modules.
    builder = instantiate(config.builder)
    routine = instantiate(config.routine)

    # Support fine-tuning mode if a pretrained model path is supplied.
    pretrained_path = config.get('pretrained_path', None)
    if pretrained_path:
        routine.load_lightning_model_state(pretrained_path)

    # Resume from last checkpoint. We assume that the checkpoint file is from
    # the end of the previous epoch. The trainer will start the next epoch.
    # Resuming from the middle of an epoch is not yet supported. See:
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/5325
    chkpt_path = Path(config_dir) / 'checkpoints' / wandb_id / 'last.ckpt' \
        if resume else None

    # Initialize the main trainer.
    callbacks = [instantiate(p) for p in config.get('callbacks', [])]
    multi_gpus = config.trainer.get('gpus', 0) > 1
    plugins = DDPPlugin(find_unused_parameters=False) if multi_gpus else None
    trainer = pl.Trainer(logger=wandb_logger,
                         callbacks=callbacks,
                         plugins=plugins,
                         weights_save_path=config_dir,
                         resume_from_checkpoint=chkpt_path,
                         enable_model_summary=False,
                         **OmegaConf.to_container(config.trainer))

    # Tuning only has an effect when either auto_scale_batch_size or
    # auto_lr_find is set to true.
    trainer.tune(routine, datamodule=builder)
    trainer.fit(routine, datamodule=builder)
    trainer.test(routine, datamodule=builder)


if __name__ == "__main__":
    app()
