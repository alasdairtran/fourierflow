import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint

from rivernet.common import Module


@Module.register('ode_decoder')
class ODEDecoder(Module):
    """
    Maps target times x_target (which we call x for consistency with NPs)
    and samples z (encoding information about the context points)
    to predictions y_target. The decoder is an ODEsolve, using torchdiffeq.
    This version contains no control.
    Models inheriting from ODEDecoder *must* either set self.xlz_to_hidden
    in constructor or override decoder_forward(). Optionally, odefunc_batch
    and	forward can also be overridden.

    Parameters
    ----------
    x_dim : int
            Dimension of x values. Currently only works for dimension of 1.
    z_dim : int
            Dimension of latent variable z. Contains [L0, z_].

    h_dim : int
            Dimension of hidden layer in odefunc.
    y_dim : int
            Dimension of y values.

    L_dim : int
            Dimension of latent state L.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_x, L_out_dim=None):
        super().__init__()

        self.x_dim = x_dim  # must be 1
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim
        if L_out_dim is None:
            L_out_dim = L_dim

        ode_layers = [nn.Linear(z_dim+x_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, L_out_dim)]

        # z = [L0, z_] so dim([L, z_, x]) = dim(z)+1
        self.latent_odefunc = nn.Sequential(*ode_layers)

        self.hidden_to_mu = nn.Linear(h_dim+L_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim+L_dim, y_dim)
        self.register_buffer(
            'initial_x', torch.FloatTensor([initial_x]).view(1, 1, 1))
        self.nfe = 0

    def odefunc_batch(self, t, v):  # v = (L(x), z_)
        self.nfe += 1
        L = v[:, :self.L_dim]
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        time = t.view(1, 1).repeat(batch_size, 1)
        vt = torch.cat((v, time), dim=1)
        dL = self.latent_odefunc(vt)
        dz_ = torch.zeros_like(z_)
        return torch.cat((dL, dz_), dim=1)

    def decoder_forward(self, x, z, latent):
        batch_size, num_points, _ = x.size()
        # compute sigma using mlp (t, L(t), z_)
        z = z[:, self.L_dim:]
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        latent_flat = latent.view(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim-self.L_dim)
        # Input is concatenation of z with every row of x
        input_triplets = torch.cat((x_flat, latent_flat, z_flat), dim=1)
        hidden = self.xlz_to_hidden(input_triplets)
        hidden = torch.cat((latent_flat, hidden), dim=1)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        return mu, pre_sigma

    # TODO: The input is always a scalar (time). So this can be made of shape
    # (batch_size, num_points)
    def forward(self, x, z):
        """
        x : torch.Tensor
                Shape (batch_size, num_points, 1)
        z : torch.Tensor
                Shape (batch_size, z_dim)
        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        self.nfe = 0
        batch_size, num_points, _ = x.size()

        # Append the initial time to the set of supplied times.
        x0 = self.initial_x.repeat(batch_size, 1, 1)
        x_sort = torch.cat((x0, x), dim=1)

        # ind specifies where each element in x ended up in times.
        times, ind = torch.unique(x_sort, sorted=True, return_inverse=True)
        # Remove the initial position index since we don't care about it.
        ind = ind[:, 1:, :]

        # Integrate forward from the batch of initial positions z.
        v = odeint(self.odefunc_batch, z, times, method='dopri5')

        # Make shape (batch_size, unique_times, z_dim).
        permuted_v = v.permute(1, 0, 2)
        latent = permuted_v[:, :, :self.L_dim]

        # Extract the relevant (latent, time) pairs for each batch.
        tiled_ind = ind.repeat(1, 1, self.L_dim)
        latent = torch.gather(latent, dim=1, index=tiled_ind)

        # here
        mu, pre_sigma = self.decoder_forward(x, z, latent)
        # end
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return mu, sigma


# Includes batching, now includes a latent state to go through MLP to get mu/sigma
@Module.register('mlp_ode_decoder')
class MLPODEDecoder(ODEDecoder):
    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_x):
        super().__init__(
            x_dim, z_dim, h_dim, y_dim, L_dim, initial_x)
        self.decode_layers = [nn.Linear(x_dim + z_dim, h_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(h_dim, h_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(h_dim, h_dim),
                              nn.ReLU(inplace=True)]
        self.xlz_to_hidden = nn.Sequential(*self.decode_layers)


@Module.register('mlp_decoder')
class MLPDecoder(Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
            Dimension of x values.

    z_dim : int
            Dimension of latent variable z.

    h_dim : int
            Dimension of hidden layer.

    y_dim : int
            Dimension of y values.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, z):
        """
        x : torch.Tensor
                Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
                Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return mu, sigma
