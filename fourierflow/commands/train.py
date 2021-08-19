import os
import shutil
import uuid
from copy import deepcopy
from glob import glob

import ptvsd
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from typer import Typer

from fourierflow.registry import Callback, Datastore, Experiment
from fourierflow.utils.parsing import yaml_to_params

app = Typer()


def get_save_dir(config_path):
    # Determine the path where the experimental results will be saved.
    parts = config_path.split('/')
    i = parts.index('experiments')
    root_dir = os.path.expandvars('$SM_MODEL_DIR')
    save_dir = os.path.join(root_dir, *parts[i+1:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def delete_old_results(results_dir, force):
    """Delete existing checkpoints and wandb logs if --force is enabled."""
    wandb_dir = os.path.join(results_dir, 'wandb')
    chkpt_dir = os.path.join(results_dir, 'checkpoints')
    if force and os.path.exists(wandb_dir):
        shutil.rmtree(wandb_dir)
    if force and os.path.exists(chkpt_dir):
        shutil.rmtree(chkpt_dir)


def upload_code_to_wandb(config_path, wandb_logger):
    """Upload all Python code for save the exact state of the experiment."""
    code_artifact = wandb.Artifact('fourierflow', type='code')
    code_artifact.add_file(config_path, 'config.yaml')
    paths = glob('fourierflow/**/*.py')
    for path in paths:
        code_artifact.add_file(path, path)
    wandb_logger.experiment.log_artifact(code_artifact)


@app.callback(invoke_without_command=True)
def main(config_path: str, overrides: str = '', force: bool = False, debug: bool = False):
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
    wandb_opts = params.pop('wandb').as_dict()
    wandb_logger = WandbLogger(save_dir=save_dir,
                               mode=os.environ.get('WANDB_MODE', 'online'),
                               config=deepcopy(params.as_dict()),
                               id=str(uuid.uuid4()),
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

    # Initialize the main trainer.
    callbacks = [Callback.from_params(p) for p in params.pop('callbacks', [])]
    multi_gpus = params.get('trainer').get('gpus', 0) > 1
    plugins = DDPPlugin(find_unused_parameters=False) if multi_gpus else None
    trainer = pl.Trainer(logger=wandb_logger,
                         callbacks=callbacks,
                         plugins=plugins,
                         weights_save_path=save_dir,
                         **params.pop('trainer').as_dict())

    # Tuning only has an effect when either auto_scale_batch_size or
    # auto_lr_find is set to true.
    trainer.tune(experiment, datamodule=datastore)
    trainer.fit(experiment, datamodule=datastore)
    trainer.test(experiment, datamodule=datastore)


if __name__ == "__main__":
    app()
