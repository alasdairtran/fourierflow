import os
import pathlib
from copy import deepcopy

import ptvsd
import pytorch_lightning as pl
import torch
import typer
import wandb
from pytorch_lightning.loggers import WandbLogger
from rivernet.datastores import Datastore
from rivernet.systems import System
from rivernet.utils.parsing import yaml_to_params

app = typer.Typer()


@app.command()
def train(config_path: str, overrides: str = '', debug: bool = False):
    """Train a model."""
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()

    parts = config_path.split('/')
    i = parts.index('experiments')
    root_dir = '.' if i == 0 else os.path.join(*parts[:i])

    params = yaml_to_params(config_path, overrides)

    save_dir = os.getenv('SM_MODEL_DIR', 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    wandb_opts = params.pop('wandb').as_dict()
    wandb_logger = WandbLogger(save_dir=save_dir,
                               mode='online',
                               config=deepcopy(params.as_dict()),
                               **wandb_opts)
    code_artifact = wandb.Artifact('rivernet', type='code')
    code_artifact.add_dir(os.path.join(root_dir, 'rivernet'))
    code_artifact.add_file(config_path, 'config.yaml')
    wandb_logger.experiment.log_artifact(code_artifact)

    datastore = Datastore.from_params(params['datastore'])
    system = System.from_params(params['system'])
    trainer = pl.Trainer(logger=wandb_logger,
                         **params.pop('trainer').as_dict())
    trainer.fit(system, datamodule=datastore)


@app.command()
def test(config_path: str, model_path: str, overrides: str = '', debug: bool = False):
    """Test a model."""
    params = yaml_to_params(config_path, overrides)
    datastore = Datastore.from_params(params['datastore'])
    system = System.from_params(params['system'], model_path=model_path)
    trainer = pl.Trainer(**params.pop('trainer').as_dict())
    trainer.test(system, datamodule=datastore)


if __name__ == "__main__":
    app()
