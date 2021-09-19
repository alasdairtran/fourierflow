import math
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import wandb

from fourierflow.builders.synthetic.ns_2d import navier_stokes_2d


def get_test_losses(dataset, groups):
    api = wandb.Api()
    outs = []
    for group in groups:
        runs = api.runs(f'alasdairtran/{dataset}', {
            'config.wandb.group': group,
            'state': 'finished'
        })
        losses = [run.summary['test_loss'] for run in runs]
        outs.append(losses)
    return np.array(outs)


def plot_line(xs, losses, ax, axis=1, **kwargs):
    means = losses.mean(axis)
    e_lower = means - losses.min(axis)
    e_upper = losses.max(axis) - means
    yerr = np.array([e_lower, e_upper])
    # yerr = losses.std(1)
    ax.errorbar(xs[:len(means)], means, yerr=yerr, **kwargs)


def plot_performance_vs_layer():
    fig = plt.figure(figsize=(3.2, 3))
    ax = plt.subplot(1, 1, 1)

    layers_1 = [4, 8, 12, 16, 20]
    layers_2 = [4, 8, 12, 16, 20, 24]
    xs = [4, 8, 12, 16, 20, 24]
    dataset = 'navier-stokes-4'

    groups = [f'zongyi/{i}_layers' for i in layers_1]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax)

    groups = [f'ablation/teaching_forcing/{i}_layers' for i in layers_1]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax)

    groups = [f'ablation/zongyi_markov/{i}_layers' for i in layers_1]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax)

    groups = [f'markov/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax)

    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Normalized MSE')
    ax.legend(['FNO', 'TF-FNO', 'M-FNO', 'F-FNO'], frameon=False)

    fig.tight_layout()
    fig.savefig('figures/loss_vs_layers.pdf')


def plot_ablation():
    fig = plt.figure(figsize=(3.2, 3))
    ax = plt.subplot(1, 1, 1)

    layers_2 = [4, 8, 12, 16, 20, 24]
    xs = [4, 8, 12, 16, 20, 24]
    dataset = 'navier-stokes-4'

    groups = [f'ablation/no_factorization/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax)

    groups = [f'ablation/no_sharing/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax)

    groups = [f'markov/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax)

    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Normalized MSE')
    ax.legend(['no factorize', 'no sharing', 'F-FNO'], frameon=False)

    fig.tight_layout()
    fig.savefig('figures/loss_vs_layers.pdf')


def get_step_losses(dataset, group):
    api = wandb.Api()
    runs = api.runs(f'alasdairtran/{dataset}', {
        'config.wandb.group': group,
        'state': 'finished'
    })
    run_losses = []
    for run in runs:
        if 'test_loss_1' in run.summary:
            losses = []
            for i in range(10):
                losses.append(run.summary[f'test_loss_{i}'])
            run_losses.append(losses)

    return np.array(run_losses)


def plot_step_loss_curves():
    fig = plt.figure(figsize=(3.2, 3))
    ax = plt.subplot(1, 1, 1)

    xs = list(range(10))
    dataset = 'navier-stokes-4'

    losses = get_step_losses(dataset, 'zongyi/4_layers')
    plot_line(xs, losses, ax, axis=0)

    losses = get_step_losses(dataset, 'ablation/zongyi_markov/4_layers')
    plot_line(xs, losses, ax, axis=0)

    losses = get_step_losses(dataset, 'markov/4_layers')
    plot_line(xs, losses, ax, axis=0)

    ax.set_xticks(xs)
    ax.set_xlabel('Step')
    ax.set_ylabel('Normalized MSE')
    ax.legend(['FNO', 'M-FNO', 'F-FNO'], frameon=False)

    fig.tight_layout()
    fig.savefig('figures/step_losses.pdf')


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
