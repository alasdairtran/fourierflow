import math

import numpy as np
import torch
import torchcde

from .base import RDataset


@RDataset.register('spiral')
class SpirialData(RDataset):
    """Spirals, some going clockwise, some going anti-clockwise.

    Parameters
    ----------
    n_samples : int
        The size of our dataset.

    n_steps : int
        How fine-grained we want the time step to be.

    seed : int
        Deterministic seed for reproducibility.

    """

    def __init__(self, n_samples=128, n_steps=100, seed=None):
        rs = np.random.RandomState(seed)

        # Generate 100 steps between 0 and 4π.
        t = np.linspace(0., 4 * math.pi, n_steps)
        # t.shape == [1, n_steps]

        # The starting time of the spiral is a random number between 0 and 2π.
        start = rs.rand(n_samples) * 2 * math.pi
        start = start[:, np.newaxis]
        # start.shape == [n_samples, 1]

        # The denominator makes the spiral smaller as time passes.
        x_pos = np.cos(start + t[np.newaxis, :]) / (1 + 0.5 * t)
        # x_pos.shape == [n_samples, n_steps]

        # Reverse the x-direction of the first half of the dataset
        x_pos[:n_samples // 2] *= -1

        y_pos = np.sin(start + t[np.newaxis, :]) / (1 + 0.5 * t)
        # y_pos.shape == [n_samples, n_steps]

        # Add some Gaussian noise
        x_pos += 0.01 * rs.randn(*x_pos.shape)
        y_pos += 0.01 * rs.randn(*y_pos.shape)

        # Neural CDEs need to be explicitly told the rate at which time passes.
        ts = t[np.newaxis, :].repeat(n_samples, axis=0)
        # ts.shape == [n_samples, n_steps]

        X = np.stack([ts, x_pos, y_pos], axis=2)
        X = X.astype(np.float32)
        # X.shape == [n_samples, n_steps, n_channels]

        # Classification labels
        y = np.zeros(n_samples, dtype=np.float32)
        y[:n_samples // 2] = 1

        # Shuffle dataset
        idx = rs.permutation(n_samples)
        self.X = torch.from_numpy(X[idx])
        self.y = torch.from_numpy(y[idx])

        # Turn data into continuous path.
        # Cache the natural cubic spline coefficients.
        self.X_coeffs = torchcde.natural_cubic_coeffs(self.X)
        # X_coeffs.shape == [n_samples, n_steps - 1, 4 * n_channels]

    def __getitem__(self, index):
        return (self.X[index], self.X_coeffs[index], self.y[index])

    def __len__(self):
        return len(self.y)
