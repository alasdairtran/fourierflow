import os
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Optional

import ptvsd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from typer import Typer

from fourierflow.registries import Datastore, Experiment
from fourierflow.utils import yaml_to_params

from .utils import get_save_dir

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
        # ptvsd doesn't play well with multiple processes.
        params['datastore']['n_workers'] = 0

    datastore = Datastore.from_params(params['datastore'])
    experiment = Experiment.from_params(params['experiment'])
    experiment.load_lightning_model_state(checkpoint_path, map_location)

    # Determine the path where the experimental test results will be saved.
    save_dir = get_save_dir(config_path)

    # We use Weights & Biases to track our experiments.
    wandb_opts = params.pop('wandb').as_dict()
    wandb_logger = WandbLogger(save_dir=save_dir,
                               mode=os.environ.get('WANDB_MODE', 'online'),
                               config=deepcopy(params.as_dict()),
                               id=datetime.now().strftime('%Y%m%d-%H%M%S-%f'),
                               **wandb_opts)

    # Start the main testing pipeline.
    trainer = pl.Trainer(logger=wandb_logger,
                         **params.pop('trainer').as_dict())
    trainer.test(experiment, datamodule=datastore)


if __name__ == "__main__":
    app()
