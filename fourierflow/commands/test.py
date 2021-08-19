import os
import uuid
from copy import deepcopy
from typing import Optional

import ptvsd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from typer import Typer

from fourierflow.registries import Datastore, Experiment
from fourierflow.utils.parsing import yaml_to_params

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: str,
         checkpoint_path: str,
         overrides: str = '',
         map_location: Optional[str] = None,
         debug: bool = False):
    """Test a Pytorch Lightning experiment."""
    params = yaml_to_params(config_path, overrides)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()

    datastore = Datastore.from_params(params['datastore'])
    experiment = Experiment.from_params(params['experiment'])
    experiment.load_lightning_model_state(checkpoint_path, map_location)

    # Determine the path where the experimental test results will be saved.
    parts = config_path.split('/')
    i = parts.index('configs')
    save_dir = os.path.expandvars('$SM_MODEL_DIR')
    results_dir = os.path.join(save_dir, *parts[i+1:-1])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # We use Weights & Biases to track our experiments.
    wandb_opts = params.pop('wandb').as_dict()
    wandb_logger = WandbLogger(save_dir=results_dir,
                               mode='online',
                               config=deepcopy(params.as_dict()),
                               id=str(uuid.uuid4()),
                               **wandb_opts)

    # Start the main testing pipeline.
    trainer = pl.Trainer(logger=wandb_logger,
                         **params.pop('trainer').as_dict())
    trainer.test(experiment, datamodule=datastore)


if __name__ == "__main__":
    app()
