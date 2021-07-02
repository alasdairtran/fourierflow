import os
from copy import deepcopy
from typing import IO, Callable, Dict, Optional, Union

import ptvsd
import pytorch_lightning as pl
import torch
import typer
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import gdown

## This is run in package root to provide defaults env vars for ${VAR} substitution
# from dotenv import load_dotenv

# # defaults for environment vars
# load_dotenv() 

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
    # When uploading artifacts, we get a lot of random "Error while calling W&B
    # API: Error 1213: Deadlock found when trying to get lock; try restarting
    # transaction (<Response [500]>)". Probably related to:
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
def test(config_path: str,
         checkpoint_path: str,
         overrides: str = '',
         map_location: Optional[str] = None,
         debug: bool = False):
    """Test a model."""
    params = yaml_to_params(config_path, overrides)
    datastore = Datastore.from_params(params['datastore'])
    experiment = Experiment.from_params(params['experiment'])
    experiment.load_lightning_model_state(checkpoint_path, map_location)

    parts = config_path.split('/')
    i = parts.index('configs')

    save_dir = os.getenv('SM_MODEL_DIR', 'results')
    results_dir = os.path.join(save_dir, *parts[i+1:-1])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    wandb_opts = params.pop('wandb').as_dict()
    wandb_logger = WandbLogger(save_dir=results_dir,
                               mode='online',
                               config=deepcopy(params.as_dict()),
                               **wandb_opts)
    trainer = pl.Trainer(logger=wandb_logger,
                         **params.pop('trainer').as_dict())
    trainer.test(experiment, datamodule=datastore)



@app.command()
def download_zongyi(config_path: str,
         checkpoint_path: str,
         overrides: str = '',
         map_location: Optional[str] = None,
         debug: bool = False):
    """Download some google datasets. 

    Should probably be in a separate module.

    Copied from a shell script:

    mkdir data/fourier && cd data/fourier
    gdown --id 16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe # Burgers_R10.zip
    gdown --id 1nzT0-Tu-LS2SoMUCcmO1qyjQd6WC9OdJ # Burgers_v100.zip
    gdown --id 1G9IW_2shmfgprPYISYt_YS8xa87p4atu # Burgers_v1000.zip
    gdown --id 1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV # Darcy_241.zip
    gdown --id 1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf # Darcy_421.zip
    gdown --id 1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d # NavierStokes_V1e-3_N5000_T50.zip
    gdown --id 1pr_Up54tNADCGhF8WLvmyTfKlCD5eEkI # NavierStokes_V1e-4_N20_T50_R256_test.zip
    gdown --id 1RmDQQ-lNdAceLXrTGY_5ErvtINIXnpl3 # NavierStokes_V1e-4_N10000_T30.zip
    gdown --id 1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5 # NavierStokes_V1e-5_N1200_T20.zip
    unzip *.zip && rm -rf *.zip
    """
    params = yaml_to_params(config_path, overrides)

    print(params)
    for hashid, fname in [
        ("16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe", "Burgers_R10.zip"),
        ("1nzT0-Tu-LS2SoMUCcmO1qyjQd6WC9OdJ", "Burgers_v100.zip"),
        ("1G9IW_2shmfgprPYISYt_YS8xa87p4atu", "Burgers_v1000.zip"),
        ("1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV", "Darcy_241.zip"),
        ("1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf", "Darcy_421.zip"),
        ("1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d", "NavierStokes_V1e-3_N5000_T50.zip"),
        ("1pr_Up54tNADCGhF8WLvmyTfKlCD5eEkI", "NavierStokes_V1e-4_N20_T50_R256_test.zip"),
        ("1RmDQQ-lNdAceLXrTGY_5ErvtINIXnpl3", "NavierStokes_V1e-4_N10000_T30.zip"),
        ("1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5", "NavierStokes_V1e-5_N1200_T20.zip"),
    ]:
        gdown.download(
            "https://drive.google.com/uc?id={hashid}".format(hashid=hashid),
            fname, postprocess=gdown.extractall)



if __name__ == "__main__":
    app()
