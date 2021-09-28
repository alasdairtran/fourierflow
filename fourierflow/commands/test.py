import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import ptvsd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from typer import Typer

from fourierflow.registries import Builder, Experiment
from fourierflow.utils import get_save_dir, yaml_to_params

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: str,
         trial: int,
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
        params['builder']['n_workers'] = 0

    # Determine the path where the experimental test results will be saved.
    save_dir = get_save_dir(config_path)

    # We use Weights & Biases to track our experiments.
    chkpt_dir = Path(save_dir) / 'checkpoints'
    paths = list(chkpt_dir.glob(f'trial-{trial}-*/*.ckpt'))
    assert len(paths) == 1
    checkpoint_path = paths[0]
    wandb_id = Path(checkpoint_path).parent.name
    trial = int(wandb_id.split('-')[1])
    params['trial'] = trial
    params['wandb']['name'] = f"{params['wandb']['group']}/{trial}"
    wandb_opts = params.get('wandb').as_dict()
    wandb_logger = WandbLogger(save_dir=save_dir,
                               mode=os.environ.get('WANDB_MODE', 'online'),
                               config=deepcopy(params.as_dict()),
                               id=wandb_id,
                               **wandb_opts)

    # Set seed for reproducibility.
    rs = np.random.RandomState(7231 + trial)
    seed = params.get('seed', rs.randint(1000, 1000000))
    pl.seed_everything(seed, workers=True)
    params['seed'] = seed

    builder = Builder.from_params(params['builder'])
    experiment = Experiment.from_params(params['experiment'])
    experiment.load_lightning_model_state(str(checkpoint_path), map_location)

    # Start the main testing pipeline.
    trainer = pl.Trainer(logger=wandb_logger,
                         **params.pop('trainer').as_dict())
    trainer.test(experiment, datamodule=builder)


if __name__ == "__main__":
    app()
