import math
import os
from timeit import default_timer

import h5py
import numpy as np
import torch
from typer import Argument, Option, Typer

from fourierflow.datastores.synthetic import GaussianRF, solve_navier_stokes_2d

app = Typer()


@app.command()
def navier_stokes(
    path: str = Argument(..., help='Path to store the generated samples'),
    n: int = Option(1200, help='Number of solutions to generate'),
    s: int = Option(256, help='Width of the solution grid'),
    t: int = Option(20, help='Final time step'),
    steps: int = Option(20, help='Number of snapshots from solution'),
    mu: float = Option(1e-5, help='Viscoity'),
    seed: int = Option(23893, help='Seed value for reproducibility'),
    delta: float = Option(1e-4, help='Internal time step for sovler'),
    batch_size: int = Option(200, help='Batch size'),
):

    device = torch.device('cuda')

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device, seed=seed)

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    ft = torch.linspace(0, 1, s+1, device=device)
    ft = ft[0:-1]

    X, Y = torch.meshgrid(ft, ft)
    f = 0.1*(torch.sin(2 * math.pi * (X + Y)) +
             torch.cos(2 * math.pi * (X + Y)))

    c = 0
    t0 = default_timer()
    data_f = h5py.File(path, 'a')
    data_f.create_dataset('a', (n, s, s), np.float32)
    data_f.create_dataset('u', (n, s, s, steps), np.float32)

    batch_size = min(n, batch_size)
    with torch.no_grad():
        for j in range(n//batch_size):
            w0 = GRF.sample(batch_size)
            sol, sol_t = solve_navier_stokes_2d(w0, f, mu, t, delta, steps)
            data_f['a'][c:(c+batch_size), ...] = w0.cpu().numpy()
            data_f['u'][c:(c+batch_size), ...] = sol.cpu().numpy()
            c += batch_size
            t1 = default_timer()
            print(j, c, t1-t0)


if __name__ == "__main__":
    app()
