import torch
import torch.nn as nn
import torch.nn.functional as F

from fourierflow.registry import Experiment, Module

from .viz import plot_deterministic_forecasts

# from torchdiffeq import odeint
# import torchcde


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

    def forward(self, coeffs):
        X = torchcde.NaturalCubicSpline(coeffs)

        X0 = X.evaluate(X.interval[0])
        # X0.shape == [batch_size, n_channels]

        # Project raw initial observation into latent space
        z0 = self.initial(X0)
        # z0.shape == [batch_size, hidden_size]

        # Actually solve the CDE.
        z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.interval)
        # Z_T.shape == [batch_size, n_steps, hidden_size]

        # Both the initial value and the terminal value are returned from
        # cdeint; extract just the terminal value, and then apply a linear map.
        z_T = z_T[:, 1]

        # Project back to observation space
        out = self.readout(z_T)

        return out


class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super().__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


@Experiment.register('cde_decoder_forecaster')
class CDEDecoderForecaster(Experiment):
    def __init__(self, n_plots):
        super().__init__()
        self.model = NeuralCDE(input_channels=2,
                               hidden_channels=128, output_channels=128)
        self.func = LatentODEfunc(128, 128)
        self.dec = Decoder(128, 1, 128)
        self.n_plots = n_plots

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        t, mu, t_x, x, x_coeffs, t_y, y = batch
        # x.shape == [batch_size, backcast_len]

        z = self.model(x_coeffs)
        # z.shape == [batch_size, hidden_size]

        times = torch.linspace(0, 1, 80).to(x.device)
        # Forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z, times, method='dopri5').permute(1, 0, 2)
        preds = self.dec(pred_z).squeeze(-1)
        mse = F.mse_loss(preds, y, reduction='mean')
        self.log('train_mse', mse)

        return mse

    def validation_step(self, batch, batch_idx):
        t, mu, t_x, x, x_coeffs, t_y, y = batch
        # x.shape == [batch_size, backcast_len]

        z = self.model(x_coeffs)
        # z.shape == [batch_size, hidden_size]

        times = torch.linspace(0, 1, 80).to(x.device)
        # Forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z, times, method='dopri5').permute(1, 0, 2)
        preds = self.dec(pred_z).squeeze(-1)
        mse = F.mse_loss(preds, y, reduction='mean')

        self.log('valid_mse', mse)

        if batch_idx == 0:
            for i in range(self.n_plots):
                e = 0 if self.global_step == 0 else self.current_epoch + 1
                name = f'e{e:02}-s{i:02}'
                plot_deterministic_forecasts(
                    self.logger.experiment, name, t[i], mu[i], t_x[i],
                    x[i], t_y[i], y[i], preds[i])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())
        return opt
