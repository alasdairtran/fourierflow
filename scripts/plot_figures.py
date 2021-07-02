import math
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

from fourierflow.datastores.synthetic.ns_2d import navier_stokes_2d


def plot_pde_inference_performance_tradeoff():
    data_path = 'data/NavierStokes_V1e-5_N1200_T20.mat'
    data = scipy.io.loadmat(os.path.expandvars(data_path))['u'].astype(np.float32)

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
    plt.show()
