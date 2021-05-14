
from math import log, pi

import torch


def fourier_encode(x, max_freq, num_bands=4, base=2):
    # Our data spans over a distance of 2. If there are 100 data points,
    # the sampling frequency (i.e. mu) is 100 / 2 = 50 Hz.
    # The Nyquist frequency is 25 Hz.
    x = x.unsqueeze(-1)
    # x.shape == [*dim_sizes, n_dims, 1]
    device, dtype, orig_x = x.device, x.dtype, x

    # max_freq is mu in the paper.
    # Create a range between (2^0 == 1) and (2^L == mu/2)
    scales = torch.logspace(0., log(max_freq / 2) / log(base),
                            num_bands, base=base, device=device, dtype=dtype)

    # Add leading dimensions
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    # scales.shape == [1, 1, 1, n_bands] for 2D images

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    # x.shape == [*dim_sizes, n_dims, n_bands * 2]

    # Interestingly enough, we also append the raw num_b ion
    x = torch.cat((x, orig_x), dim=-1)
    # x.shape == [*dim_sizes, n_dims, n_bands * 2 + 1]
    return x
