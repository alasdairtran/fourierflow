import math

from torch.optim.lr_scheduler import LambdaLR

from fourierflow.registries.schedulers import Scheduler


class LinearLRLambda:
    def __init__(self, num_warmup_steps, num_training_steps):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    def __call__(self, current_step):  # the function formerly known as "bar"
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(0.0, float(self.num_training_steps - current_step) /
                   float(max(1, self.num_training_steps - self.num_warmup_steps)))


class LinearWithWarmupScheduler(Scheduler, LambdaLR):
    def __init__(self, optimizer, num_warmup_steps: int,
                 num_training_steps: int, last_epoch=-1, verbose=False):

        lr_lambda = LinearLRLambda(num_warmup_steps, num_training_steps)
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)
