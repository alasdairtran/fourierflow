from dotenv import load_dotenv  # isort:skip
load_dotenv()  # noqa

import hydra
from omegaconf import OmegaConf

import fourierflow.builders
import fourierflow.callbacks
import fourierflow.modules
import fourierflow.routines
import fourierflow.schedulers
from fourierflow.utils import import_string

# Allow partial instantiations of optimizers and schedulers.
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
# Allow importing constants
OmegaConf.register_new_resolver("import", import_string)
# Remove hack once this is resolved: https://github.com/omry/omegaconf/issues/91
OmegaConf.register_new_resolver("eval", eval)
