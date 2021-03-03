import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_sines(device, t, y, mu, nodep, expt):
    t_c = []
    x_c = []

    rs = np.random.RandomState(5214)  # always select same random targets

    length_context_1 = 1
    length_context_2 = 14
    start_point = 2  # third time step
    darkness = 0.15  # transparency of sampled path
    truth_darkness = 1  # transparency of true path
    colour = '#9013FE'  # purple-ish
    number_to_plot = 35  # possible paths
    x_min, x_max = -math.pi, math.pi
    y_min, y_max = -1.1, 1.1

    t_c.append(t[start_point])
    x_c.append(y[start_point])
    choices = np.arange(start_point, len(y))

    for i in range(length_context_1-1):
        choice = rs.choice(choices, replace=False)
        t_c.append(t[choice])
        x_c.append(y[choice])

    t_context_1 = torch.FloatTensor(t_c).float().reshape(
        (1, length_context_1, 1)).to(device)
    x_context_1 = torch.FloatTensor(x_c).float().reshape(
        (1, length_context_1, 1)).to(device)

    for num_points in range(length_context_2):
        choice = rs.choice(choices, replace=False)
        t_c.append(t[choice])
        x_c.append(y[choice])

    t_context_2 = torch.FloatTensor(t_c).float().reshape(
        1, length_context_1+length_context_2, 1).to(device)
    x_context_2 = torch.FloatTensor(x_c).float().reshape(
        1, length_context_1+length_context_2, 1).to(device)
    t_target = t.reshape((1, len(t), 1)).to(device)

    # making the plot
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(number_to_plot):
        # Neural process returns distribution over y_target
        p_y_pred = nodep(t_context_1, x_context_1, t_target)
        # Extract mean of distribution
        pred_mu = p_y_pred.loc.detach()
        ax.plot(t_target.cpu().numpy()[0], pred_mu.cpu().numpy()[0],
                alpha=darkness, c=colour, zorder=-number_to_plot)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.plot(t, mu, c='k', linestyle='--',
            alpha=truth_darkness, zorder=1)
    ax.scatter(t_context_1[0].cpu().numpy(), x_context_1[0].cpu().numpy(),
               c='k', alpha=truth_darkness, zorder=2)
    ax.scatter(t, y, c='r', s=3, alpha=darkness, zorder=0)
    fig.tight_layout()
    expt.log_figure(figure=fig, figure_name='one')

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(number_to_plot):
        # Neural process returns distribution over y_target
        p_y_pred = nodep(t_context_2, x_context_2, t_target)
        # Extract mean of distribution
        pred_mu = p_y_pred.loc.detach()
        ax.plot(t_target.cpu().numpy()[0], pred_mu.cpu().numpy()[0],
                alpha=darkness, c=colour, zorder=-number_to_plot)
    ax.set_xlabel('t', fontsize=16)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.plot(t, mu, c='k', linestyle='--',
            alpha=truth_darkness, zorder=1)
    ax.scatter(t_context_2[0].cpu().numpy(), x_context_2[0].cpu().numpy(),
               c='k', alpha=truth_darkness, zorder=2)
    ax.scatter(t, y, c='r', s=3, alpha=darkness, zorder=0)
    expt.log_figure(figure=fig, figure_name='many')
