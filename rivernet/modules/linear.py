import logging
import math

import torch.nn as nn

logger = logging.getLogger(__name__)


class GehringLinear(nn.Linear):
    """A linear layer with Gehring initialization and weight normalization."""

    def __init__(self, in_features, out_features, dropout=0, bias=True,
                 weight_norm=True):
        self.dropout = dropout
        self.weight_norm = weight_norm
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        # One problem with initialization from the uniform distribution is that
        # the distribution of the outputs has a variance that grows with the
        # number of inputs. It turns out that we can normalize the variance of
        # each neuron’s output to 1 by scaling its weight vector by the square
        # root of its fan-in (i.e. its number of inputs). Dropout further
        # increases the variance of each input, so we need to scale down std.
        # See A.3. in Gehring et al (2017): https://arxiv.org/pdf/1705.03122.
        std = math.sqrt((1 - self.dropout) / self.in_features)
        self.weight.data.normal_(mean=0, std=std)
        if self.bias is not None:
            self.bias.data.fill_(0)

        # Weight normalization is a reparameterization that decouples the
        # magnitude of a weight tensor from its direction. See Salimans and
        # Kingma (2016): https://arxiv.org/abs/1602.07868.
        if self.weight_norm:
            nn.utils.weight_norm(self)
