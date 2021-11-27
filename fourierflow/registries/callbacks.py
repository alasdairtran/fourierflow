from allennlp.common import Registrable
from pytorch_lightning.callbacks import Callback as LightingCallback
from pytorch_lightning.callbacks import (EarlyStopping, GPUStatsMonitor,
                                         GradientAccumulationScheduler,
                                         LearningRateMonitor, ModelSummary)

from fourierflow.callbacks import (CustomModelCheckpoint,
                                   StochasticWeightAveraging)


class Callback(Registrable, LightingCallback):
    ...


Registrable._registry[Callback] = {
    'early_stopping': (EarlyStopping, None),
    'gpu_stats_monitor': (GPUStatsMonitor, None),
    'gradient_accumulation_scheduler': (GradientAccumulationScheduler, None),
    'learning_rate_monitor': (LearningRateMonitor, None),
    'model_checkpoint': (CustomModelCheckpoint, None),
    'model_summary': (ModelSummary, None),
    'stochastic_weight_averaging': (StochasticWeightAveraging, None),
}
