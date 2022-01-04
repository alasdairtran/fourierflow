from pathlib import Path
from typing import List, Optional

import hydra
import ptvsd
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typer import Argument, Typer

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: Path,
         overrides: Optional[List[str]] = Argument(None),
         debug: bool = False):
    """Train a JAX experiment."""
    config_dir = config_path.parent
    config_name = config_path.stem
    hydra.initialize(config_path=str(Path('../..') / config_dir))
    config = hydra.compose(config_name, overrides=overrides or [])
    OmegaConf.set_struct(config, False)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()

    # Initialize the dataset and experiment modules.
    builder = instantiate(config.builder)
    routine = instantiate(config.routine)

    routine.fit(
        inputs=builder.train_dataloader(),
        epochs=2,
        steps_per_epoch=200,
        batch_size=1,
        validation_data=builder.val_dataloader(),
        shuffle=True,
        # callbacks=[eg.callbacks.TensorBoard("summaries")]
    )


if __name__ == "__main__":
    app()
