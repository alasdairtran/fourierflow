import pickle
from pathlib import Path
from typing import List, Optional

import debugpy
import hydra
import jax
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typer import Argument, Typer

from fourierflow.utils import import_string

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: Path,
         overrides: Optional[List[str]] = Argument(None),
         trial: int = 0,
         map_location: Optional[str] = None,
         debug: bool = False):
    """Test a Pytorch Lightning experiment."""
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
        config.builder.num_workers = 0
        jax.config.update('jax_disable_jit', True)

    # Strange bug: We need to check if cuda is availabe first; otherwise,
    # sometimes lightning's CUDAAccelerator.is_available() returns false :-/
    torch.cuda.is_available()

    chkpt_dir = Path(config_dir) / 'checkpoints'
    paths = list(chkpt_dir.glob(f'trial-{trial}-*/epoch*.ckpt'))
    assert len(paths) == 1
    checkpoint_path = paths[0]
    config.trial = trial
    config.wandb.name = f"{config.wandb.group}/{trial}"

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

    # Start the main testing pipeline.
    Trainer = import_string(config.trainer.pop(
        '_target_', 'pytorch_lightning.Trainer'))

    trainer = Trainer(logger=False, enable_checkpointing=False,
                          **OmegaConf.to_container(config.trainer))
    trainer.test(routine, datamodule=builder)

    routine = routine.cuda()
    loader = builder.test_dataloader()
    with torch.no_grad():
        for batch in loader:
            pred = routine.forward(batch).cpu().numpy()
            out_path = Path(config_dir) / 'sample.pkl'
            with open(out_path, 'wb') as f:
                pickle.dump([batch, pred], f)
            break


if __name__ == "__main__":
    app()
