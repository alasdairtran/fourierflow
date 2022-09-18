import os
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, cast

import debugpy
import hydra
import jax
import numpy as np
import pytorch_lightning as pl
import scipy.io
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from typer import Argument, Typer

from fourierflow.builders.synthetic.ns_2d import solve_navier_stokes_2d

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: Optional[Path] = Argument(None),
         overrides: Optional[List[str]] = Argument(None),
         trial: Optional[int] = None,
         map_location: Optional[str] = None,
         debug: bool = False):
    """Test a Pytorch Lightning experiment."""

    # Calculate inference time for Crank-Nicolson 2nd order method.
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

    config_dir = config_path.parent
    config_name = config_path.stem
    hydra.initialize(config_path=Path('../..') /
                     config_dir, version_base='1.2')
    config = hydra.compose(config_name, overrides=overrides)
    OmegaConf.set_struct(config, False)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        debugpy.listen(5678)
        debugpy.wait_for_client()
        # debugger doesn't play well with multiple processes.
        config.builder.n_workers = 0
        jax.config.update('jax_disable_jit', True)

    # We use Weights & Biases to track our experiments.
    chkpt_dir = Path(config_dir) / 'checkpoints'
    paths = list(chkpt_dir.glob(f'trial-{trial}-*/epoch*.ckpt'))
    # if len(paths) > 1:
    #     paths = list(chkpt_dir.glob(f'trial-{trial}-*/last.ckpt'))
    assert len(paths) == 1
    checkpoint_path = paths[0]
    wandb_id = checkpoint_path.parent.name
    trial = int(wandb_id.split('-')[1])
    config.trial = trial
    config.wandb.name = f"{config.wandb.group}/{trial}"
    wandb_opts = cast(dict, OmegaConf.to_container(config.wandb))
    wandb_logger = WandbLogger(save_dir=str(config_dir),
                               mode=os.environ.get('WANDB_MODE', 'offline'),
                               config=deepcopy(OmegaConf.to_container(config)),
                               id=wandb_id,
                               **wandb_opts)

    # Set seed for reproducibility.
    rs = np.random.RandomState(7231 + trial)
    seed = config.get('seed', rs.randint(1000, 1000000))
    pl.seed_everything(seed, workers=True)
    config.seed = seed
    if 'seed' in config.trainer:
        config.trainer.seed = seed

    builder = instantiate(config.builder)
    routine = instantiate(config.routine)
    routine.load_lightning_model_state(str(checkpoint_path), map_location)

    batch = builder.inference_data()
    if batch is not None:
        T = batch['data'].shape[-1]
        n_steps = routine.n_steps or (T - 1)
        routine = routine.cuda()
        batch = routine.convert_data(batch)
        routine.warmup()

        start = time.time()
        routine.infer(batch)
        elapsed = time.time() - start

        elapsed /= len(batch['data'])
        elapsed /= routine.step_size * n_steps
        wandb_logger.experiment.log({'inference_time': elapsed})


if __name__ == "__main__":
    app()
