from pathlib import Path
from typing import List, Optional

import hydra
import ptvsd
from omegaconf import OmegaConf
from typer import Argument, Typer

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_dir: str,
         overrides: Optional[List[str]] = Argument(None),
         debug: bool = False):
    """Train a JAX experiment."""
    hydra.initialize(config_path=Path('../..') / config_dir)
    config = hydra.compose(config_name='config', overrides=overrides)
    OmegaConf.set_struct(config, False)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()


if __name__ == "__main__":
    app()
