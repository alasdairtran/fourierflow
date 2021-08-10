from allennlp.common import Registrable
from pytorch_lightning.callbacks import Callback as LightingCallback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


class Callback(Registrable, LightingCallback):
    ...


Registrable._registry[Callback] = {
    'model_checkpoint': (ModelCheckpoint, None),
    'learning_rate_monitor': (LearningRateMonitor, None),
}
