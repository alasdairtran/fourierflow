import math
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

from fourierflow.builders.synthetic.ns_2d import navier_stokes_2d


def plot_performance_vs_layers():
    xs = [4, 8, 12, 16, 20, 24]
    ys_zongyi = [0.1381, 0.1544, 0.1684, 0.1762, 0.1837, np.nan]
    ys_teacher = [0.1299, 0.1094, 0.1176, 0.144, 0.2177, np.nan]
    ys_ours = [0.05705, 0.03422, 0.02861, 0.02613, 0.02408, 0.02287]

    fig = plt.figure(figsize=(9, 4))
    ax = plt.subplot(1, 2, 1)
    ax.errorbar(xs, ys_zongyi, marker='o')
    ax.errorbar(xs, ys_ours, marker='x')
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.errorbar(xs, ys_teacher, marker='.')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Normalized MSE')
    ax.legend(['FNO', 'CW-FNO', 'TF-FNO'], frameon=False)

    fig.tight_layout()
    fig.savefig('figures/loss_vs_layers.pdf')


def plot_pde_inference_performance_tradeoff():
    data_path = 'data/NavierStokes_V1e-5_N1200_T20.mat'
    data = scipy.io.loadmat(os.path.expandvars(data_path))[
        'u'].astype(np.float32)

    w0 = data[:, :, :, 10]

    device = torch.device('cuda')
    s = 64

    t = torch.linspace(0, 1, s+1, device=device)
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t)
    f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

    record_steps = 20
    start = time.time()
    sol, sol_t = navier_stokes_2d(w0, f, 1e-3, 10.0, 1e-4, record_steps)
    elapsed = time.time() - start

    x = [0.08692677319049835, 0.0646510198712349,
         0.059111643582582474, 0.056166261434555054, 0]
    y = [10, 17, 26, 46, elapsed]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    ax.scatter(x, y)
    ax.set_xlabel('error')
    ax.set_ylabel('inference time')
    ax.set_title('PDE Inference Performance Tradeoff')
