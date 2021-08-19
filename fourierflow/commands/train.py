import os
import shutil
from copy import deepcopy
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional

import ptvsd
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from typer import Typer

from fourierflow.registries import Callback, Datastore, Experiment
from fourierflow.utils.parsing import yaml_to_params

from .utils import get_save_dir

app = Typer()


def delete_old_results(results_dir, force):
    """Delete existing checkpoints and wandb logs if --force is enabled."""
    wandb_dir = Path(results_dir) / 'wandb'
    chkpt_dir = Path(results_dir) / 'checkpoints'
    if force and wandb_dir.exists():
        shutil.rmtree(wandb_dir)
    if force and chkpt_dir.exists():
        shutil.rmtree(chkpt_dir)


def upload_code_to_wandb(config_path, wandb_logger):
    """Upload all Python code for save the exact state of the experiment."""
    code_artifact = wandb.Artifact('fourierflow', type='code')
    code_artifact.add_file(config_path, 'config.yaml')
    paths = glob('fourierflow/**/*.py')
    for path in paths:
        code_artifact.add_file(path, path)
    wandb_logger.experiment.log_artifact(code_artifact)


def get_experiment_id(checkpoint_id, save_dir, resume):
    chkpt_dir = Path(save_dir) / 'checkpoints'
    if resume and not checkpoint_id and chkpt_dir.exists:
        paths = chkpt_dir.glob('*/last.ckpt')
        checkpoint_id = next(paths).parent.name
    return checkpoint_id or datetime.now().strftime('%Y%m%d-%H%M%S-%f')


@app.callback(invoke_without_command=True)
def main(config_path: str, overrides: str = '', force: bool = False,
         resume: bool = False, checkpoint_id: Optional[str] = None,
         debug: bool = False):
    """Train a Pytorch Lightning experiment."""
    params = yaml_to_params(config_path, overrides)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()
        # ptvsd doesn't play well with multiple processes.
        params['datastore']['n_workers'] = 0

    # Set up directories to save experimental outputs.
    save_dir = get_save_dir(config_path)
    delete_old_results(save_dir, force)

    # We use Weights & Biases to track our experiments.
    wandb_id = get_experiment_id(checkpoint_id, save_dir, resume)
    wandb_opts = params.pop('wandb').as_dict()
    wandb_logger = WandbLogger(save_dir=save_dir,
                               mode=os.environ.get('WANDB_MODE', 'online'),
                               config=deepcopy(params.as_dict()),
                               id=wandb_id,
                               **wandb_opts)

    # Set seed and upload code for reproducibility.
    seed = params.get('seed', 38124)
    pl.seed_everything(seed, workers=True)
    upload_code_to_wandb(config_path, wandb_logger)

    # Initialize the dataset and experiment modules.
    datastore = Datastore.from_params(params['datastore'])
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
    trainer.tune(experiment, datamodule=datastore)
    trainer.fit(experiment, datamodule=datastore)
    trainer.test(experiment, datamodule=datastore)


if __name__ == "__main__":
    app()
