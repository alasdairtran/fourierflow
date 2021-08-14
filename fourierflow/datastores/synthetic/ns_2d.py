import math

import numpy as np
import torch
from einops import rearrange, repeat
from tqdm import tqdm


# w0: initial vorticity
# f: forcing term
#visc: viscosity (1/Re)
# T: final time
# delta_t: internal time-step for solve (descrease if blow-up)
# record_steps: number of in-time snapshots to record
def solve_navier_stokes_2d(w0, f, visc, T, delta_t, record_steps):

    # Grid size - must be power of 2
    N = w0.shape[-1]

    # Maximum frequency
    k_max = math.floor(N / 2)

    # Number of steps to final time
    steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1, 2], norm='backward')

    # Forcing to Fourier space
    f_h = torch.fft.fftn(f, dim=[-2, -1], norm='backward')

    # If same forcing for the whole batch
    if len(f_h.shape) < len(w_h.shape):
        f_h = rearrange(f_h, '... -> 1 ...')

    # Record solution every this number of steps
    record_time = math.floor(steps / record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((
        torch.arange(start=0, end=k_max, step=1, device=w0.device),
        torch.arange(start=-k_max, end=0, step=1, device=w0.device)),
        0).repeat(N, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi**2) * (k_x**2 + k_y**2)
    lap[0, 0] = 1.0

    if isinstance(visc, np.ndarray):
        visc = torch.from_numpy(visc).to(w0.device)
        visc = repeat(visc, 'b -> b m n', m=N, n=N)
        lap = repeat(lap, 'm n -> b m n', b=w0.shape[0])

    # Dealiasing mask
    dealias = torch.unsqueeze(
        torch.logical_and(
            torch.abs(k_y) <= (2.0 / 3.0) * k_max,
            torch.abs(k_x) <= (2.0 / 3.0) * k_max
        ).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in tqdm(range(steps)):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        q_real_temp = q.real.clone()
        q.real = -2 * math.pi * k_y * q.imag
        q.imag = 2 * math.pi * k_y * q_real_temp
        q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        v_real_temp = v.real.clone()
        v.real = 2 * math.pi * k_x * v.imag
        v.imag = -2 * math.pi * k_x * v_real_temp
        v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

        # Partial x of vorticity
        w_x = w_h.clone()
        w_x_temp = w_x.real.clone()
        w_x.real = -2 * math.pi * k_x * w_x.imag
        w_x.imag = 2 * math.pi * k_x * w_x_temp
        w_x = torch.fft.ifftn(w_x, dim=[1, 2], norm='backward').real

        # Partial y of vorticity
        w_y = w_h.clone()
        w_y_temp = w_y.real.clone()
        w_y.real = -2 * math.pi * k_y * w_y.imag
        w_y.imag = 2 * math.pi * k_y * w_y_temp
        w_y = torch.fft.ifftn(w_y, dim=[1, 2], norm='backward').real

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.fftn(q * w_x + v * w_y,
                             dim=[1, 2], norm='backward')

        # Dealias
        F_h *= dealias

        # Cranck-Nicholson update
        factor = 0.5 * delta_t * visc * lap
        num = -delta_t * F_h + delta_t * f_h + (1.0 - factor) * w_h
        w_h = num / (1.0 + factor)

        # Update real time (used only for recording)
        t += delta_t

        if (j + 1) % record_time == 0:
            # Solution in physical space
            w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real
            if w.isnan().any().item():
                raise ValueError('NaN values found.')

            # Record solution and time
            sol[..., c] = w
            sol_t[c] = t

            c += 1

    return sol, sol_t
