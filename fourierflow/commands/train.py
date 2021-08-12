import os
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


@app.callback(invoke_without_command=True)
def main(config_path: str, overrides: str = '', debug: bool = False):
    """Train a Pytorch Lightning experiment."""
    params = yaml_to_params(config_path, overrides)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()

    # Determine the path where the experimental results will be saved.
    parts = config_path.split('/')
    i = parts.index('configs')
    save_dir = os.path.expandvars('$SM_MODEL_DIR')
    results_dir = os.path.join(save_dir, *parts[i+1:-1])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # We use Weights & Biases to track our experiments.
    wandb_opts = params.pop('wandb').as_dict()
    wandb_logger = WandbLogger(save_dir=results_dir,
                               mode=os.environ.get('WANDB_MODE', 'online'),
                               config=deepcopy(params.as_dict()),
                               id=str(uuid.uuid4()),
                               **wandb_opts)

    # To ensure reproduciblity, we seed the whole Pytorch Lightning pipeline.
    seed = params.get('seed', '38124')
    os.environ['PL_GLOBAL_SEED'] = seed
    os.environ['PL_SEED_WORKERS'] = '1'

    # Upload all Python code for save the exact state of the experiment code.
    code_artifact = wandb.Artifact('fourierflow', type='code')
    code_artifact.add_file(config_path, 'config.yaml')
    paths = glob('fourierflow/**/*.py')
    for path in paths:
        code_artifact.add_file(path, path)
    wandb_logger.experiment.log_artifact(code_artifact)

    # ptvsd doesn't play well with multiple processes.
    if debug:
        params['datastore']['n_workers'] = 0
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
                         **params.pop('trainer').as_dict())

    # Tuning only has an effect when either auto_scale_batch_size or
    # auto_lr_find is set to true.
    trainer.tune(experiment, datamodule=datastore)
    trainer.fit(experiment, datamodule=datastore)
    trainer.test(experiment, datamodule=datastore)


if __name__ == "__main__":
    app()
