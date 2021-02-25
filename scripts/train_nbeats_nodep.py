import math
import pickle

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint
from torchdyn.models import NeuralDE


class GehringLinear(nn.Linear):
    """A linear layer with Gehring initialization and weight normalization."""

    def __init__(self, in_features, out_features, dropout=0, bias=True,
                 weight_norm=True):
        self.dropout = dropout
        self.weight_norm = weight_norm
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        std = math.sqrt((1 - self.dropout) / self.in_features)
        self.weight.data.normal_(mean=0, std=std)
        if self.bias is not None:
            self.bias.data.fill_(0)

        if self.weight_norm:
            nn.utils.weight_norm(self)


class VevoDataset(Dataset):
    def __init__(self, split):
        data_path = '/localdata/u4921817/projects/phd/radflow/data/vevo/vevo_static.hdf5'
        views = h5py.File(data_path, 'r')['views'][...]

        if split == 'train':
            with open('/localdata/u4921817/projects/phd/radflow/data/vevo/vevo_all_nodes.pkl', 'rb') as f:
                indices = sorted(pickle.load(f))
                views = views[indices]
        else:
            # if True:
            with open('/localdata/u4921817/projects/phd/radflow/data/vevo/vevo_static_connected_nodes.pkl', 'rb') as f:
                indices = sorted(pickle.load(f))
                views = views[indices]

        # Forward-fill missing values
        mask = views == -1
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        views = views[np.arange(idx.shape[0])[:, None], idx]

        self.samples = torch.log1p(torch.from_numpy(views).float())
        if split == 'train':
            self.samples = self.samples[:, :49]
        elif split == 'valid':
            self.samples = self.samples[:, 7:56]
        elif split == 'test':
            self.samples = self.samples[:, 14:63]
        else:
            raise ValueError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TimeSeriesODE(pl.LightningModule):
    def __init__(self,
                 forecast_length=7,
                 backcast_length=42,
                 hidden_size=256,
                 latent_size=256,
                 l_size=64,
                 dropout=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.l_size = l_size
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.nfe = 0

        self.encoder = nn.Sequential(
            GehringLinear(2, hidden_size),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            GehringLinear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            GehringLinear(hidden_size, latent_size),
        )

        self.mean_proj = nn.Sequential(
            GehringLinear(latent_size, latent_size),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            GehringLinear(latent_size, latent_size),
        )

        self.std_proj = GehringLinear(latent_size, latent_size)

        self.decoder = nn.Sequential(
            GehringLinear(latent_size + 1, hidden_size),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            GehringLinear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            GehringLinear(hidden_size, hidden_size),
        )
        self.hidden_to_mu = GehringLinear(hidden_size + l_size, 1)
        self.hidden_to_sigma = GehringLinear(hidden_size + l_size, 1)

        self.ode_f = nn.Sequential(
            GehringLinear(latent_size + 1, hidden_size),
            nn.Tanh(),
            # nn.Dropout(dropout),
            GehringLinear(hidden_size, hidden_size),
            nn.Tanh(),
            # nn.Dropout(dropout),
            GehringLinear(hidden_size, l_size),
        )

        # self.model = NeuralDE(ode_f, sensitivity='adjoint', solver='dopri5')
        # self.model = NeuralDE(ode_f, sensitivity='autograd', solver='rk4')

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def odefunc_batch(self, t, v):  # v = (L(x), z_)
        self.nfe += 1
        batch_size = v.shape[0]
        # t.shape == [1]
        # v.shape == [batch_size, latent_size]

        # Split random context into an initial latent position and a sub-context
        L, z_ = v[:, :self.l_size], v[:, self.l_size:]
        time = t.view(1, 1).repeat(batch_size, 1)
        # time.shape == [batch_size, 1]

        # Add time information to the
        vt = torch.cat((v, time), dim=1)
        dL = self.ode_f(vt)
        dz_ = torch.zeros_like(z_)

        return torch.cat((dL, dz_), dim=1)

    def xy_to_mu_sigma(self, x, y):
        X = torch.stack([x, y], dim=2)
        # X.shape == [batch_size, total_len, 2]

        # Obtain a representation for each context point
        X = self.encoder(X)
        # X.shape == [batch_size, total_len, latent_size]

        # Global latent context parameters
        G = X.mean(dim=1)
        # G.shape == [batch_size, latent_size]

        # Obtain the mean and standard deviation parameters
        mu = self.mean_proj(G)
        # mu.shape == [batch_size, latent_size]

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.std_proj(G))
        # sigma.shape == [batch_size, latent_size]

        return mu, sigma

    def decoder_forward(self, x, z, latent):
        # x.shape == [batch_size, seq_len, 1]
        # z.shape == [batch_size, latent_size]
        # latent.shape == [batch_size, seq_len, l_size]

        B, T, _ = x.shape

        # compute sigma using mlp (t, L(t), z_)
        z = z[:, self.l_size:]
        # z.shape == [batch_size, l_size]

        z = z.unsqueeze(1).repeat(1, T, 1)
        # z.shape == [batch_size, seq_len, l_size]

        # Input is concatenation of z with every row of x
        input_triplets = torch.cat((x, latent, z), dim=2)
        # input_triplets.shape == [batch_size, seq_len, 1 + latent_size]

        hidden = self.decoder(input_triplets)
        # hidden.shape == [batch_size, seq_len, hidden_size]

        hidden = torch.cat((latent, hidden), dim=2)
        mu = self.hidden_to_mu(hidden).squeeze(-1)
        pre_sigma = self.hidden_to_sigma(hidden).squeeze(-1)
        # mu.shape == pre_sigma.shape == [batch_size, seq_len]

        return mu, pre_sigma

    def xz_to_y(self, x, z):
        # z.shape == [batch_size, latent_size]
        self.nfe = 0

        x = x.unsqueeze(2)
        # x.shape == [batch_size, seq_len, 1]

        x0 = -x[0, 1].clone()
        x0 = x0.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
        # x0.shape == [batch_size, 1, 1]

        # Append the initial time to the set of supplied times.
        x_sort = torch.cat([x0, x], dim=1)
        # x_sort.shape == [batch_size, seq_len + 1, 1]

        # ind specifies where each element in x ended up in times.
        # times will be a flattened array
        times, inv_idx = torch.unique(x_sort, sorted=True, return_inverse=True)
        # times.shape == [n_steps]
        # inv_idx.shape == x_sort.shape

        # Remove the initial position index since we don't care about it.
        inv_idx = inv_idx[:, 1:, :]
        # inv_idx.shape == [batch_size, seq_len, 1]

        # Integrate forward from the batch of initial positions z.
        v = odeint(self.odefunc_batch, z, times, method='dopri5')
        # v.shape == [n_steps, batch_size, latent_size]

        # Make shape (batch_size, unique_times, z_dim).
        v = v.permute(1, 0, 2)
        # v.shape == [batch_size, n_steps, latent_size]

        latent = v[:, :, :self.l_size]
        # v.shape == [batch_size, n_steps, l_size]

        # Extract the relevant (latent, time) pairs for each batch.
        tiled_ind = inv_idx.repeat(1, 1, self.l_size)
        # inv_idx.shape == [batch_size, seq_len, l_size]

        latent = torch.gather(latent, dim=1, index=tiled_ind)
        # latent.shape == [batch_size, seq_len, l_size]

        mu, pre_sigma = self.decoder_forward(x, z, latent)
        # mu.shape == pre_sigma.shape == [batch_size, seq_len]

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)

        return mu, sigma

    def get_split(self, batch):
        S = self.backcast_length
        L = self.forecast_length
        T = S + L
        # We follow the convention given in "Empirical Evaluation of Neural
        # Process Objectives" where context is a subset of target points. This was
        # shown to work best empirically.
        x_target = torch.linspace(0, 1, T).unsqueeze(0).expand_as(batch)
        x_target = x_target.to(batch.device)
        # x_target.shape == [batch_size, total_len]

        x_context = x_target[:, :S]
        # x_context.shape == [batch_size, backcast_len]

        y_target = batch
        # y_target.shape == [batch_size, total_len]

        y_context = batch[:, :S]
        # y_context.shape == [batch_size, backcast_len]

        return x_context, y_context, x_target, y_target

    def training_step(self, batch, batch_idx):
        x_context, y_context, x_target, y_target = self.get_split(batch)
        # batch.shape == [batch_size, total_len]

        # Encode target and context (context needs to be encoded to
        # calculate kl term)
        mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
        mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)

        # Sample from encoded distribution using reparameterization trick
        q_target = Normal(mu_target, sigma_target)
        q_context = Normal(mu_context, sigma_context)
        z_sample = q_target.rsample()
        # z_sample.shape == [batch_size, latent_size]

        # Get parameters of output distribution
        y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
        py_pred = Normal(y_pred_mu, y_pred_sigma)

        loss = self.get_loss(py_pred, y_target, q_target, q_context)

        # Calculate SMAPE
        forecasts = torch.exp(y_pred_mu) - 1
        targets = torch.exp(batch) - 1
        smape = self.get_smape(forecasts, targets)

        loss += smape

        self.log('train_smape', smape)
        self.log('train_loss', loss)

        return loss

    def get_loss(self, py_pred, y_target, q_target, q_context):
        # Maximizing the log-likelihood is the first term of the loss.
        log_likelihood = py_pred.log_prob(y_target)
        # log_likelihood.shape == [batch_size, total_len]

        # Take the mean over batch and sum over the sequence
        log_likelihood = log_likelihood.mean(dim=0).sum()
        # log_likelihood.shape = [1]

        # KL has shape (batch_size, r_dim).
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        # kl.shape == [batch_size, latent_size]

        # Take mean over batch and sum over latent dim (dimension of normal distribution)
        kl = kl.mean(dim=0).sum()
        # kl.shape = [1]

        loss = -log_likelihood + kl
        return loss

    def get_smape(self, forecasts, targets):
        denominator = torch.abs(forecasts) + torch.abs(targets)
        smape = 200 * torch.abs(forecasts - targets) / denominator
        # smape = torch.nan_to_num(smape, nan=200, posinf=200, neginf=200)
        smape[torch.isnan(smape)] = 0
        smape = smape.mean()
        return smape

    def validation_step(self, batch, batch_idx):
        x_context, y_context, x_target, y_target = self.get_split(batch)
        # batch.shape == [batch_size, total_len]

        # Encode only context (no access to target during inference)
        mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)

        # Sample from encoded distribution using reparameterization trick
        q_context = Normal(mu_context, sigma_context)
        z_sample = q_context.rsample()
        # z_sample.shape == [batch_size, latent_size]

        # Get parameters of output distribution
        y_pred_mu, _ = self.xz_to_y(x_target, z_sample)

        # Calculate SMAPE
        forecasts = torch.exp(y_pred_mu[:, -self.forecast_length:]) - 1
        targets = torch.exp(batch[:, -self.forecast_length:]) - 1
        smape = self.get_smape(forecasts, targets) * len(batch)

        return {'val_smape': smape, 'batch_size': len(batch)}

    def validation_epoch_end(self, outputs):
        val_smape_mean = 0
        count = 0

        for output in outputs:
            val_smape = output['val_smape']
            val_smape_mean += val_smape
            count += output['batch_size']
        val_smape_mean /= (count)
        print('test smape', val_smape_mean)
        self.log('val_smape', val_smape_mean)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.nll_loss(logits, y)
    #     self.log('test_loss', loss)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)

        num_warmup_steps = 500
        num_training_steps = 950 * 40  # 215 * 40
        last_epoch = -1
        num_cycles = 0.5

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / \
                float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        sched = {'scheduler': torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch),
                 'monitor': 'loss',
                 'interval': 'step',
                 'frequency': 10}
        return [opt], [sched]


def main():
    train_loader = DataLoader(VevoDataset('train'),
                              batch_size=64,
                              shuffle=True,
                              num_workers=4)

    valid_loader = DataLoader(VevoDataset('valid'),
                              batch_size=64, num_workers=4)
    test_loader = DataLoader(VevoDataset('test'), batch_size=64, num_workers=4)

    ts_ode = TimeSeriesODE()
    trainer = pl.Trainer(gpus=1, max_epochs=40, gradient_clip_val=0.1)
    trainer.fit(ts_ode, train_loader, test_loader)

    # result = trainer.test(ts_ode, test_loader)


if __name__ == '__main__':
    main()
