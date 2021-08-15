import math
import os
from enum import Enum

import h5py
import numpy as np
import ptvsd
import torch
from einops import repeat
from typer import Argument, Option, Typer

from fourierflow.datastores.synthetic import GaussianRF, solve_navier_stokes_2d

app = Typer()


class Force(str, Enum):
    li = "li"
    random = "random"


def get_random_force(b, s, device, cycles, scaling):
    ft = torch.linspace(0, 1, s+1).to(device)
    ft = ft[0:-1]
    X, Y = torch.meshgrid(ft, ft)
    X = repeat(X, 'x y -> b x y', b=b)
    Y = repeat(Y, 'x y -> b x y', b=b)

    f = 0
    for p in range(1, cycles + 1):
        k = 2 * math.pi * p
        f += torch.rand(b, 1, 1).to(device) * torch.sin(k * X)
        f += torch.rand(b, 1, 1).to(device) * torch.cos(k * X)

        f += torch.rand(b, 1, 1).to(device) * torch.sin(k * Y)
        f += torch.rand(b, 1, 1).to(device) * torch.cos(k * Y)

        f += torch.rand(b, 1, 1).to(device) * torch.sin(k * (X + Y))
        f += torch.rand(b, 1, 1).to(device) * torch.cos(k * (X + Y))

    f = f * scaling

    return f


@app.command()
def navier_stokes(
    path: str = Argument(..., help='Path to store the generated samples'),
    n_train: int = Option(1000, help='Number of train solutions to generate'),
    n_valid: int = Option(200, help='Number of valid solutions to generate'),
    n_test: int = Option(200, help='Number of test solutions to generate'),
    s: int = Option(256, help='Width of the solution grid'),
    t: int = Option(20, help='Final time step'),
    steps: int = Option(20, help='Number of snapshots from solution'),
    mu: float = Option(1e-5, help='Viscoity'),
    mu_min: float = Option(1e-5, help='Minimium viscoity'),
    mu_max: float = Option(1e-5, help='Maximum viscoity'),
    seed: int = Option(23893, help='Seed value for reproducibility'),
    delta: float = Option(1e-4, help='Internal time step for sovler'),
    batch_size: int = Option(100, help='Batch size'),
    force: Force = Option(Force.li, help='Type of forcing function'),
    cycles: int = Option(2, help='Number of cycles in forcing function'),
    scaling: float = Option(0.1, help='Scaling of forcing function'),
    debug: bool = Option(False, help='Enable debugging mode with ptvsd'),
):
    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        ptvsd.enable_attach(address=('0.0.0.0', 5678))
        ptvsd.wait_for_attach()

    device = torch.device('cuda')
    torch.manual_seed(seed)
    np.random.seed(seed + 1234)

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    if force == Force.li:
        # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
        ft = torch.linspace(0, 1, s+1, device=device)
        ft = ft[0:-1]
        X, Y = torch.meshgrid(ft, ft)
        f = 0.1*(torch.sin(2 * math.pi * (X + Y)) +
                 torch.cos(2 * math.pi * (X + Y)))

    data_f = h5py.File(path, 'a')

    def generate_split(n, split):
        print('Generating split:', split)
        data_f.create_dataset(f'{split}/a', (n, s, s), np.float32)
        data_f.create_dataset(f'{split}/f', (n, s, s), np.float32)
        data_f.create_dataset(f'{split}/u', (n, s, s, steps), np.float32)
        data_f.create_dataset(f'{split}/mu', (n,), np.float32)
        b = min(n, batch_size)
        c = 0

        with torch.no_grad():
            for j in range(n // b):
                print('batch', j)
                w0 = GRF.sample(b)

                if force == Force.random:
                    f = get_random_force(b, s, device, cycles, scaling)

                if mu_min != mu_max:
                    mu = np.random.rand(b) * (mu_max - mu_min) + mu_min

                sol, _ = solve_navier_stokes_2d(w0, f, mu, t, delta, steps)
                data_f[f'{split}/a'][c:(c+b), ...] = w0.cpu().numpy()
                data_f[f'{split}/u'][c:(c+b), ...] = sol

                if force == Force.random:
                    data_f[f'{split}/f'][c:(c+b), ...] = f.cpu().numpy()

                data_f[f'{split}/mu'][c:(c+b)] = mu

                c += b

    generate_split(n_train, 'train')
    generate_split(n_valid, 'valid')
    generate_split(n_test, 'test')


if __name__ == "__main__":
    app()
