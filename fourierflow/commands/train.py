import os
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, cast

import hydra
import jax
import numpy as np
import ptvsd
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from typer import Argument, Typer

from fourierflow.utils import (delete_old_results, get_experiment_id,
                               import_string, upload_code_to_wandb)

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: Path,
         overrides: Optional[List[str]] = Argument(None),
         force: bool = False,
         resume: bool = False,
         checkpoint_id: Optional[str] = None,
         trial: int = 0,
         debug: bool = False,
         no_logging: bool = False):
    """Train a Pytorch Lightning experiment."""
    config_dir = config_path.parent
    config_name = config_path.stem
    hydra.initialize(config_path=Path('../..') / config_dir)
    config = hydra.compose(config_name, overrides=overrides)
    OmegaConf.set_struct(config, False)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()
        # ptvsd doesn't play well with multiple processes.
        config.builder.num_workers = 0
        jax.config.update('jax_disable_jit', True)
        # jax.config.update("jax_debug_nans", True)

    # Set up directories to save experimental outputs.
    delete_old_results(config_dir, force, trial, resume)

    # Set seed for reproducibility.
    rs = np.random.RandomState(7231 + trial)
    seed = config.get('seed', rs.randint(1000, 1000000))
    pl.seed_everything(seed, workers=True)
    config.seed = seed
    wandb_id = get_experiment_id(checkpoint_id, trial, config_dir, resume)
    config.trial = trial
    if 'seed' in config.trainer:
        config.trainer.seed = seed

    # Initialize the dataset and experiment modules.
    builder = instantiate(config.builder)
    routine = instantiate(config.routine)

    # Support fine-tuning mode if a pretrained model path is supplied.
    pretrained_path = config.get('pretrained_path', None)
    if pretrained_path:
        routine.load_lightning_model_state(pretrained_path)

    # Resume from last checkpoint. We assume that the checkpoint file is from
    # the end of the previous epoch. The trainer will start the next epoch.
    # Resuming from the middle of an epoch is not yet supported. See:
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/5325
    chkpt_path = Path(config_dir) / 'checkpoints' / wandb_id / 'last.ckpt' \
        if resume else None

    # Initialize the main trainer.
    callbacks = [instantiate(p) for p in config.get('callbacks', [])]
    multi_gpus = config.trainer.get('gpus', 0) > 1
    plugins = DDPPlugin(find_unused_parameters=False) if multi_gpus else None
    if no_logging:
        logger = False
        enable_checkpointing = False
        callbacks = []
    else:
        # We use Weights & Biases to track our experiments.
        config.wandb.name = f"{config.wandb.group}/{trial}"
        wandb_opts = cast(dict, OmegaConf.to_container(config.wandb))
        logger = WandbLogger(save_dir=str(config_dir),
                             mode=os.environ.get('WANDB_MODE', 'offline'),
                             config=deepcopy(OmegaConf.to_container(config)),
                             id=wandb_id,
                             **wandb_opts)
        upload_code_to_wandb(Path(config_dir) / 'config.yaml', logger)
        enable_checkpointing = True
        c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
        c.cleanup(wandb.util.from_human_size("100GB"))

    Trainer = import_string(config.trainer.pop(
        '_target_', 'pytorch_lightning.Trainer'))
    trainer = Trainer(logger=logger,
                      enable_checkpointing=enable_checkpointing,
                      callbacks=callbacks,
                      plugins=plugins,
                      weights_save_path=config_dir,
                      resume_from_checkpoint=chkpt_path,
                      enable_model_summary=False,
                      **OmegaConf.to_container(config.trainer))

    # Tuning only has an effect when either auto_scale_batch_size or
    # auto_lr_find is set to true.
    trainer.tune(routine, datamodule=builder)
    trainer.fit(routine, datamodule=builder)

    # Load best checkpoint before testing.
    chkpt_dir = Path(config_dir) / 'checkpoints'
    paths = list(chkpt_dir.glob(f'trial-{trial}-*/epoch*.ckpt'))
    assert len(paths) == 1
    checkpoint_path = paths[0]
    routine.load_lightning_model_state(str(checkpoint_path))
    trainer.test(routine, datamodule=builder)

    # Compute inference time
    if logger:
        batch = builder.inference_data()
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
        logger.experiment.log({'inference_time': elapsed})


if __name__ == "__main__":
    app()
