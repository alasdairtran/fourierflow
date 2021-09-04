import os
import shutil
from copy import deepcopy
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import ptvsd
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from typer import Typer

from fourierflow.registries import Callback, Builder, Experiment
from fourierflow.utils import ExistingExperimentFound, yaml_to_params

from .utils import get_save_dir

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


def get_experiment_id(checkpoint_id, trial, save_dir, resume):
    chkpt_dir = Path(save_dir) / 'checkpoints'
    if resume and not checkpoint_id and chkpt_dir.exists:
        paths = chkpt_dir.glob('*/last.ckpt')
        checkpoint_id = next(paths).parent.name
    now = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    return checkpoint_id or f'trial-{trial}-{now}'


@app.callback(invoke_without_command=True)
def main(config_path: str, overrides: str = '', force: bool = False,
         resume: bool = False, checkpoint_id: Optional[str] = None,
         trial: int = 0, debug: bool = False):
    """Train a Pytorch Lightning experiment."""
    params = yaml_to_params(config_path, overrides)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()
        # ptvsd doesn't play well with multiple processes.
        params['builder']['n_workers'] = 0

    # Set up directories to save experimental outputs.
    save_dir = get_save_dir(config_path)
    delete_old_results(save_dir, force, trial, resume)

    # Set seed for reproducibility.
    rs = np.random.RandomState(7231 + trial)
    seed = params.get('seed', rs.randint(1000, 1000000))
    pl.seed_everything(seed, workers=True)
    params['seed'] = seed

    # We use Weights & Biases to track our experiments.
    wandb_id = get_experiment_id(checkpoint_id, trial, save_dir, resume)
    params['trial'] = trial
    params['wandb']['name'] = f"{params['wandb']['group']}/{trial}"
    wandb_opts = params.get('wandb').as_dict()
    wandb_logger = WandbLogger(save_dir=save_dir,
                               mode=os.environ.get('WANDB_MODE', 'online'),
                               config=deepcopy(params.as_dict()),
                               id=wandb_id,
                               **wandb_opts)
    upload_code_to_wandb(config_path, wandb_logger)

    # Initialize the dataset and experiment modules.
    builder = Builder.from_params(params['builder'])
    experiment = Experiment.from_params(params['experiment'])

    # Support fine-tuning mode if a pretrained model path is supplied.
    pretrained_path = params.pop('pretrained_path', None)
    if pretrained_path:
        experiment.load_lightning_model_state(pretrained_path)

    # Resume from last checkpoint. We assume that the checkpoint file is from
    # the end of the previous epoch. The trainer will start the next epoch.
    # Resuming from the middle of an epoch is not yet supported. See:
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/5325
    chkpt_path = Path(save_dir) / 'checkpoints' / wandb_id / 'last.ckpt' \
        if resume else None

    # Initialize the main trainer.
    callbacks = [Callback.from_params(p) for p in params.pop('callbacks', [])]
    multi_gpus = params.get('trainer').get('gpus', 0) > 1
    plugins = DDPPlugin(find_unused_parameters=False) if multi_gpus else None
    trainer = pl.Trainer(logger=wandb_logger,
                         callbacks=callbacks,
                         plugins=plugins,
                         weights_save_path=save_dir,
                         resume_from_checkpoint=chkpt_path,
                         **params.pop('trainer').as_dict())

    # Tuning only has an effect when either auto_scale_batch_size or
    # auto_lr_find is set to true.
    trainer.tune(experiment, datamodule=builder)
    trainer.fit(experiment, datamodule=builder)
    trainer.test(experiment, datamodule=builder)


if __name__ == "__main__":
    app()
