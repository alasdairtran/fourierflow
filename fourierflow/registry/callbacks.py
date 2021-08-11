from allennlp.common import Registrable
from pytorch_lightning.callbacks import Callback as LightingCallback
from pytorch_lightning.callbacks import (EarlyStopping, GPUStatsMonitor,
                                         GradientAccumulationScheduler,
                                         LearningRateMonitor, ModelCheckpoint)

from fourierflow.callbacks import StochasticWeightAveraging


class Callback(Registrable, LightingCallback):
    ...


Registrable._registry[Callback] = {
    'early_stopping': (EarlyStopping, None),
    'gpu_stats_monitor': (GPUStatsMonitor, None),
    'gradient_accumulation_scheduler': (GradientAccumulationScheduler, None),
    'learning_rate_monitor': (LearningRateMonitor, None),
    'model_checkpoint': (ModelCheckpoint, None),
    'stochastic_weight_averaging': (StochasticWeightAveraging, None),
}
