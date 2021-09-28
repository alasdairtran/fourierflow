import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import ptvsd
import pytorch_lightning as pl
import scipy.io
import torch
from pytorch_lightning.loggers import WandbLogger
from typer import Argument, Typer

from fourierflow.builders.synthetic.ns_2d import solve_navier_stokes_2d
from fourierflow.registries import Experiment
from fourierflow.utils import get_save_dir, yaml_to_params

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: Optional[str] = Argument(None),
         trial: Optional[int] = None,
         overrides: str = '',
         map_location: Optional[str] = None,
         debug: bool = False):
    """Test a Pytorch Lightning experiment."""
    if not config_path:
        data_path = 'data/fourier/NavierStokes_V1e-5_N1200_T20.mat'
        data = scipy.io.loadmat(data_path)['u'].astype(np.float32)
        w0 = data[:512, :, :, 10]
        w0 = torch.from_numpy(w0).cuda()

        start = time.time()
        sol = solve_navier_stokes_2d(w0=w0, visc=1e-5, T=10, delta_t=1e-4,
                                     record_steps=10, force='li')
        elasped = time.time() - start
        print(elasped)
        return

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
    wandb_id = checkpoint_path.parent.name
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

    # builder = Builder.from_params(params['builder'])
    experiment = Experiment.from_params(params['experiment'])
    experiment.load_lightning_model_state(str(checkpoint_path), map_location)

    data_path = 'data/fourier/NavierStokes_V1e-5_N1200_T20.mat'
    data = scipy.io.loadmat(data_path)['u'].astype(np.float32)[:512]
    data = torch.from_numpy(data).cuda()
    experiment = experiment.cuda()
    start = time.time()
    with torch.no_grad():
        experiment(data)
    elasped = time.time() - start
    wandb_logger.experiment.log({'inference_time': elasped})


if __name__ == "__main__":
    app()
