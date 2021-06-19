import math
from timeit import default_timer

import h5py
import torch
from tqdm import tqdm

from .random_fields import GaussianRF

# from drawnow import drawnow, figure


# w0: initial vorticity
# f: forcing term
#visc: viscosity (1/Re)
# T: final time
# delta_t: internal time-step for solve (descrease if blow-up)
# record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):

    # Grid size - must be power of 2
    N = w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N/2.0)

    # Number of steps to final time
    steps = math.ceil(T/delta_t)

    # Initial vorticity to Fourier space
#     w_h = torch.rfft(w0, 2, normalized=False, onesided=False)
    w_h = torch.fft.fftn(w0, dim=[1, 2], norm='backward')

    # Forcing to Fourier space
#     f_h = torch.rfft(f, 2, normalized=False, onesided=False)
    f_h = torch.fft.fftn(f, dim=[0, 1], norm='backward')

    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(
        start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N, 1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)
    # Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0, 0] = 1.0
    # Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (
        2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in tqdm(range(steps)):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h.real = psi_h.real/lap
        psi_h.imag = psi_h.imag/lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q.real.clone()
        q.real = -2*math.pi*k_y*q.imag
        q.imag = 2*math.pi*k_y*temp
#         q = torch.irfft(q, 2, normalized=False,
#                         onesided=False, signal_sizes=(N, N))
        q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v.real.clone()
        v.real = 2*math.pi*k_x*v.imag
        v.imag = -2*math.pi*k_x*temp
#         v = torch.irfft(v, 2, normalized=False,
#                         onesided=False, signal_sizes=(N, N))
        v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

        # Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x.real.clone()
        w_x.real = -2*math.pi*k_x*w_x.imag
        w_x.imag = 2*math.pi*k_x*temp
#         w_x = torch.irfft(w_x, 2, normalized=False,
#                           onesided=False, signal_sizes=(N, N))
        w_x = torch.fft.ifftn(w_x, dim=[1, 2], norm='backward').real

        # Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y.real.clone()
        w_y.real = -2*math.pi*k_y*w_y.imag
        w_y.imag = 2*math.pi*k_y*temp
#         w_y = torch.irfft(w_y, 2, normalized=False,
#                           onesided=False, signal_sizes=(N, N))
        w_y = torch.fft.ifftn(w_y, dim=[1, 2], norm='backward').real

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
#         F_h = torch.rfft(q*w_x + v*w_y, 2, normalized=False, onesided=False)
        F_h = torch.fft.fftn(q*w_x + v*w_y, dim=[1, 2], norm='backward')

        # Dealias
        F_h.real = dealias * F_h.real
        F_h.imag = dealias * F_h.imag

        # Cranck-Nicholson update
        w_h.real = (-delta_t*F_h.real + delta_t*f_h.real + (1.0 -
                                                            0.5*delta_t*visc*lap)*w_h.real)/(1.0 + 0.5*delta_t*visc*lap)
        w_h.imag = (-delta_t*F_h[..., 1] + delta_t*f_h.imag + (1.0 -
                                                               0.5*delta_t*visc*lap)*w_h.imag)/(1.0 + 0.5*delta_t*visc*lap)

        # Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            # Solution in physical space
            #             w = torch.irfft(w_h, 2, normalized=False,
            #                             onesided=False, signal_sizes=(N, N))
            w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real

            # Record solution and time
            sol[..., c] = w
            sol_t[c] = t

            c += 1

    return sol, sol_t


device = torch.device('cuda')

# Resolution
s = 256
sub = 1

# Number of solutions to generate
N = 1200

# Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s+1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t)
f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

# Number of snapshots from solution
record_steps = 50

# Inputs
a = torch.zeros(N, s, s)
# Solutions
u = torch.zeros(N, s, s, record_steps)

# Solve equations in batches (order of magnitude speed-up)

# Batch size
bsize = 100

c = 0
t0 = default_timer()
viscosity = 1e-6
data_path = 'ns_data_1e-6.h5'
data_f = h5py.File(data_path, 'a')
data_f.create_dataset('a', (N, s, s), np.float32)
data_f.create_dataset('u', (N, s, s, record_steps), np.float32)

for j in range(N//bsize):

    # Sample random feilds
    w0 = GRF.sample(bsize)

    # Solve NS
    sol, sol_t = navier_stokes_2d(w0, f, viscosity, 50.0, 1e-4, record_steps)

    data_f['a'][c:(c+bsize), ...] = w0.cpu().numpy()
    data_f['u'][c:(c+bsize), ...] = sol.cpu().numpy()

    c += bsize
    t1 = default_timer()
    print(j, c, t1-t0)
