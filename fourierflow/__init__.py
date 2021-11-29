from dotenv import load_dotenv  # isort:skip
load_dotenv()  # noqa

import hydra
from omegaconf import OmegaConf

import fourierflow.builders
import fourierflow.callbacks
import fourierflow.experiments
import fourierflow.modules
import fourierflow.registries
import fourierflow.schedulers

# Allow partial instantiations of optimizers and schedulers.
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
