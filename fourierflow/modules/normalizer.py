import torch
import torch.nn as nn
from einops import rearrange


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
        self.dim_sizes = None

    def _accumulate(self, x):
        x_count = x.shape[0]
        x_sum = x.sum(dim=0)
        x_sum_squared = (x**2).sum(dim=0)

        self.sum += x_sum
        self.sum_squared += x_sum_squared
        self.count += x_count
        self.n_accumulations += 1

    def _pool_dims(self, x):
        _, *dim_sizes, _ = x.shape
        self.dim_sizes = dim_sizes
        if self.dim_sizes:
            x = rearrange(x, 'b ... h -> (b ...) h')

        return x

    def _unpool_dims(self, x):
        if len(self.dim_sizes) == 1:
            x = rearrange(x, '(b m) h -> b m h', m=self.dim_sizes[0])
        elif len(self.dim_sizes) == 2:
            m, n = self.dim_sizes
            x = rearrange(x, '(b m n) h -> b m n h', m=m, n=n)
        return x

    def forward(self, x):
        x = self._pool_dims(x)
        # x.shape == [batch_size, latent_dim]

        if self.training and self.n_accumulations < self.max_accumulations:
            self._accumulate(x)

        x = (x - self.mean) / self.std
        x = self._unpool_dims(x)

        return x

    def inverse(self, x, channel=None):
        x = self._pool_dims(x)

        if channel is None:
            x = x * self.std + self.mean
        else:
            x = x * self.std[channel] + self.mean[channel]

        x = self._unpool_dims(x)

        return x

    @property
    def mean(self):
        safe_count = max(self.count, self.one)
        return self.sum / safe_count

    @property
    def std(self):
        safe_count = max(self.count, self.one)
        std = torch.sqrt(self.sum_squared / safe_count - self.mean**2)
        return torch.maximum(std, self.std_epsilon)
