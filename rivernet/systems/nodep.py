
import numpy as np
import torch
from torch.distributions.kl import kl_divergence

from rivernet.modules import NeuralODEProcess

from .base import System
from .viz import plot_sines


@System.register('time_series_ode')
class TimeSeriesODE(System):
    def __init__(self,
                 x_dim=1,
                 y_dim=1,
                 r_dim=128,
                 z_dim=128,
                 h_dim=128,
                 L_dim=32,
                 initial_x=-3.2,
                 forecast_length=7,
                 backcast_length=42,
                 hidden_size=256,
                 latent_size=256,
                 l_size=64,
                 dropout=0.1,
                 num_context_range=(1, 20),
                 extra_target_range=(0, 5),
                 testing_context_size=10):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.l_size = l_size
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.nfe = 0
        self.num_context_range = num_context_range
        self.num_extra_target_range = extra_target_range
        self.test_context_size = testing_context_size
        self.rs = np.random.RandomState(1242)

        initial_x = torch.FloatTensor([initial_x]).view(1, 1, 1)
        self.nodep = NeuralODEProcess(
            x_dim, y_dim, r_dim, z_dim, h_dim, L_dim, initial_x)

        # Fix a random datapoint for plotting
        self.t = None
        self.y = None
        self.mu = None

        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y, mu = batch

        # Sample number of context and target points
        num_context = self.rs.randint(*self.num_context_range)
        num_extra_target = self.rs.randint(*self.num_extra_target_range)

        x_context, y_context, x_target, y_target = \
            context_target_split(x, y, num_context, num_extra_target)

        p_y_pred, q_target, q_context = \
            self.nodep(x_context, y_context, x_target, y_target)

        loss = self._loss(p_y_pred, y_target, q_target, q_context)

        self.log('train_loss', loss)

        return loss

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
                Distribution over y output by Neural Process.

        y_target : torch.Tensor
                Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
                Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
                Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl

    def validation_step(self, batch, batch_idx):
        if self.t is None:
            self.t, self.y, self.mu = batch[0][0], batch[0][1], batch[0][2]

        x, y, mu = batch
        x_context, y_context, x_target, y_target = \
            context_target_split(x, y, self.test_context_size, 0)

        p_y_pred = self.nodep(x_context, y_context, x_target, y_target)

        epoch_logp_loss = -p_y_pred.log_prob(
            y_target).mean(dim=0).sum().item()
        output = p_y_pred.loc.detach()
        mse = torch.mean((y_target-output)**2)
        epoch_mse_loss = mse.item()

        self.log('valid_mse', epoch_mse_loss)
        self.log('valid_logp', epoch_logp_loss)

    def validation_epoch_end(self, outputs):
        device = next(self.parameters()).device
        plot_sines(device, self.t, self.y, self.mu,
                   self.nodep, self.logger.experiment)

    def configure_optimizers(self):
        opt = torch.optim.RMSprop(self.parameters(), lr=1e-3)
        return opt


def context_target_split(x, y, num_context, num_extra_target, locations=None):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
            Shape (batch_size, num_points, y_dim)

    num_context : int
            Number of context points.

    num_extra_target : int
            Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    if locations is None:
        locations = np.random.choice(num_points,
                                     size=num_context + num_extra_target,
                                     replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target


def context_target_split_full(x, y, num_context, num_extra_target, locations=None):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
            Shape (batch_size, num_points, y_dim)

    num_context : int
            Number of context points.

    num_extra_target : int
            Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    if locations is None:
        locations = np.random.choice(2 * num_points // 3,
                                     size=num_context,
                                     replace=False)
    x_context = x[:, locations, :]
    y_context = y[:, locations, :]
    x_target = x.clone()
    y_target = y.clone()
    return x_context, y_context, x_target, y_target
