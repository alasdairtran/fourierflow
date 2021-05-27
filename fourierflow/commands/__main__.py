import os
from copy import deepcopy

import ptvsd
import pytorch_lightning as pl
import typer
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from fourierflow.common import Datastore, Experiment
from fourierflow.utils.parsing import yaml_to_params

app = typer.Typer()


@app.command()
def train(config_path: str, overrides: str = '', debug: bool = False):
    """Train a model."""
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()

    parts = config_path.split('/')
    i = parts.index('configs')
    root_dir = '.' if i == 0 else os.path.join(*parts[:i])

    params = yaml_to_params(config_path, overrides)

    save_dir = os.getenv('SM_MODEL_DIR', 'results')
    results_dir = os.path.join(save_dir, *parts[i+1:-1])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    checkpoint_params = params.pop('checkpointer')
    metric = checkpoint_params.pop('validation_metric')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, 'checkpoints'),
        monitor=metric)

    wandb_opts = params.pop('wandb').as_dict()
    wandb_logger = WandbLogger(save_dir=results_dir,
                               mode='online',
                               config=deepcopy(params.as_dict()),
                               **wandb_opts)
    # code_artifact = wandb.Artifact('fourierflow', type='code')
    # code_artifact.add_dir(os.path.join(root_dir, 'fourierflow'))
    # code_artifact.add_file(config_path, 'config.yaml')
    # wandb_logger.experiment.log_artifact(code_artifact)

    datastore = Datastore.from_params(params['datastore'])
    experiment = Experiment.from_params(params['experiment'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(logger=wandb_logger,
                         callbacks=[lr_monitor, checkpoint_callback],
                         **params.pop('trainer').as_dict())
    trainer.fit(experiment, datamodule=datastore)
    trainer.test(experiment, datamodule=datastore)


@app.command()
def test(config_path: str, model_path: str = None, overrides: str = '', debug: bool = False):
    """Test a model."""
    params = yaml_to_params(config_path, overrides)
    datastore = Datastore.from_params(params['datastore'])
    experiment = Experiment.from_params(
        params['experiment'], model_path=model_path)
    # from fourierflow.experiments import Fourier2DExperiment
    # experiment = Fourier2DExperiment.load_from_checkpoint(model_path)
    trainer = pl.Trainer(**params.pop('trainer').as_dict())
    trainer.test(experiment, datamodule=datastore)


if __name__ == "__main__":
    app()
