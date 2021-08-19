import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8):
        super().__init__()
        self.max_accumulations = max_accumulations
        self.register_buffer('count', torch.tensor(0.0))
        self.register_buffer('n_accumulations', torch.tensor(0.0))
        self.register_buffer('sum', torch.full(size, 0.0))
        self.register_buffer('sum_squared', torch.full(size, 0.0))
        self.register_buffer('one', torch.tensor(1.0))
        self.register_buffer('std_epsilon', torch.full(size, std_epsilon))

    def _accumulate(self, x):
        x_count = x.shape[0]
        x_sum = x.sum(dim=0)
        x_sum_squared = (x**2).sum(dim=0)

        self.sum += x_sum
        self.sum_squared += x_sum_squared
        self.count += x_count
        self.n_accumulations += 1

    def forward(self, x):
        # x.shape == [batch_size, latent_dim]
        if self.training and self.n_accumulations < self.max_accumulations:
            self._accumulate(x)

        return (x - self._mean) / self._std

    def inverse(self, x):
        return x * self._std + self._mean

    @property
    def _mean(self):
        safe_count = max(self.count, self.one)
        return self.sum / safe_count

    @property
    def _std(self):
        safe_count = max(self.count, self.one)
        std = torch.sqrt(self.sum_squared / safe_count - self.mean**2)
        return torch.maximum(std, self.std_epsilon)
