import torch
import torch.nn.functional as F
import torchcde

from .base import System


######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
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

        # Initial hidden state should be a function of the first observation.
        # X_interval returns [start_time, end_time]
        # Usually start_time = 0 and end_time = n_steps - 1
        # X0 should simply return the raw observation at t = 0
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
        pred_y = self.readout(z_T)

        return pred_y


@System.register('cde_classification')
class CDEClassification(System):
    def __init__(self):
        super().__init__()
        self.model = NeuralCDE(input_channels=3,
                               hidden_channels=8, output_channels=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, X_coeffs, y = batch
        pred_y = self.model(X_coeffs).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred_y, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        X, X_coeffs, y = batch
        pred_y = self.model(X_coeffs).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred_y, y)

        binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(y.dtype)
        prediction_matches = (binary_prediction == y).to(y.dtype)
        proportion_correct = prediction_matches.sum() / y.size(0)

        self.log('valid_prop_correct', proportion_correct)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())
        return opt
