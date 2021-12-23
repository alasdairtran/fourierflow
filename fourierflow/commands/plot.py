import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import seaborn as sns
import wandb
from matplotlib.lines import Line2D
from typer import Typer

from fourierflow.viz.heatmap import MidpointNormalize

pal = sns.color_palette()
app = Typer()


@app.command()
def layer():
    fig = plt.figure(figsize=(8, 2.6))

    ax = plt.subplot(1, 3, 1)
    lines_1 = plot_performance_vs_layer(ax)

    ax = plt.subplot(1, 3, 2)
    lines_2 = plot_ablation(ax)

    ax = plt.subplot(1, 3, 3)
    plot_step_loss_curves(ax)

    lines = [lines_1[1], lines_1[0], lines_1[2]] + lines_2
    labels = ['FNO (proposed by Li et al. [2021a])',
              'FNO-TF (with teacher forcing)',
              'FNO-M (with Markov assumption)',
              'FNO++ (with a bag of tricks)',
              'F-FNO-NW (without weight sharing)',
              'F-FNO (our full model)']
    lgd = fig.legend(handles=lines,
                     labels=labels,
                     loc="center",
                     borderaxespad=0.1,
                     bbox_to_anchor=[1.2, 0.55])

    fig.tight_layout()
    fig.savefig('figures/performance.pdf',
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')


@app.command()
def complexity():
    fig = plt.figure(figsize=(8, 2.6))

    ax = plt.subplot(1, 3, 1)
    lines_1 = plot_parameters(ax)

    ax = plt.subplot(1, 3, 2)
    lines_2 = plot_pde_training_performance_tradeoff(ax)

    ax = plt.subplot(1, 3, 3)
    plot_pde_inference_performance_tradeoff(ax)

    sim_line = Line2D(range(1), range(1), color="white",
                      marker='o', markerfacecolor=pal[4])
    lines = [sim_line] + lines_2[-1:] + lines_1
    labels = ['Crank–Nicolson numerical simulator',
              'FNO (proposed by Li et al. [2021a])',
              'FNO++ (with a bag of tricks)',
              'F-FNO-NW (without weight sharing)',
              'F-FNO (our full model)']
    lgd = fig.legend(handles=lines,
                     labels=labels,
                     loc="center",
                     borderaxespad=0.1,
                     bbox_to_anchor=[1.2, 0.55])

    fig.tight_layout()
    fig.savefig('figures/complexity.pdf',
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')


@app.command()
def heatmaps():
    data_path = './data/ns_contextual/ns_time_varying_forces.h5'
    h5f = h5py.File(data_path)

    plot_heatmap(h5f['train']['f'][897, ..., 50], cmap='PuOr',
                 vmin=-0.7, vmax=0.7, out_path='figures/f50.svg')

    plot_heatmap(h5f['train']['f'][897, ..., 100], cmap='PuOr',
                 vmin=-0.7, vmax=0.7, out_path='figures/f100.svg')

    plot_heatmap(h5f['train']['u'][897, ..., 50], cmap='RdBu',
                 vmin=-3, vmax=3, out_path='figures/w50.svg')

    plot_heatmap(h5f['train']['u'][897, ..., 100], cmap='RdBu',
                 vmin=-3, vmax=3, out_path='figures/w100.svg')

    plot_heatmap(h5f['train']['u'][897, ..., 150], cmap='RdBu',
                 vmin=-3, vmax=3, out_path='figures/w150.svg')


@app.command()
def table_3():
    dataset = 'navier-stokes-4'
    layers_1 = [4, 8, 12, 16, 20]
    layers_2 = [4, 8, 12, 16, 20, 24]

    groups = [f'zongyi/{i}_layers' for i in layers_1]
    get_summary(dataset, groups)
    print('\\midrule')

    groups = [f'ablation/teaching_forcing/{i}_layers' for i in layers_1]
    get_summary(dataset, groups)
    print('\\midrule')

    groups = [f'ablation/zongyi_markov/{i}_layers' for i in layers_1]
    get_summary(dataset, groups)
    print('\\midrule')

    groups = [f'ablation/no_factorization/{i}_layers' for i in layers_2]
    get_summary(dataset, groups)
    print('\\midrule')

    groups = [f'ablation/no_sharing/{i}_layers' for i in layers_2]
    get_summary(dataset, groups)
    print('\\midrule')

    groups = [f'markov/{i}_layers' for i in layers_2]
    get_summary(dataset, groups)


@app.command()
def flow(i: int = 0):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)

    data_path = './data/zongyi/NavierStokes_V1e-5_N1200_T20.mat'
    data = scipy.io.loadmat(data_path)['u'].astype(np.float32)

    ims = []
    for t in range(20):
        im = ax.imshow(data[i, ..., t], cmap='RdBu', interpolation='bilinear')
        ax.set_axis_off()
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                    repeat_delay=500)

    ani.save("demo.gif", writer='imagemagick')


@app.command()
def poster():
    plot_scalable()
    plot_10_steps()
    plot_poster_pde_inference()


def get_summary(dataset, groups):
    names = {
        ('zongyi',): 'FNO (reproduced)',
        ('ablation', 'no_factorization'): 'FNO++ (with bags of tricks)',
        ('ablation', 'teaching_forcing'): 'FNO-TF (with teacher forcing)',
        ('ablation', 'zongyi_markov'): 'FNO-M (with Markov assumption)',
        ('ablation', 'no_sharing'): 'F-FNO-NW (without weight sharing)',
        ('markov',): 'F-FNO (our full model)',

    }
    api = wandb.Api()

    parts = groups[0].split('/')
    g = names[tuple(parts[:-1])]
    print(f'\multirow{{5}}{{*}}{{{g}}}')

    for group in groups:
        runs = api.runs(f'alasdairtran/{dataset}', {
            'config.wandb.group': group,
            'state': 'finished'
        })
        losses = [run.summary['test_loss'] for run in runs]
        losses = np.array(losses) * 100
        assert len(losses) == 3

        train_times = [run.summary['_runtime'] for run in runs]
        train_times = np.array(train_times) / 3600
        assert len(train_times) == 3

        test_times = [run.summary['inference_time']
                      for run in runs if 'inference_time' in run.summary]
        test_times = np.array(test_times)
        assert len(test_times) == 3

        params = [run.summary['n_params'] for run in runs]

        mean = losses.mean()
        std = losses.std()
        train_t = train_times.mean()
        test_t = test_times.mean()
        parts = group.split('/')
        layers = int(parts[-1].split('_')[0])
        print(f' & {layers} & {params[0]:,} & ${mean:.2f} \pm {std:.2f}$ & '
              f'{train_t:.1f} & {test_t:.1f} \\\\')


def plot_heatmap(array,  cmap, vmin, vmax, out_path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    ax.imshow(array, cmap=cmap, norm=norm)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')


def get_test_losses(dataset, groups):
    api = wandb.Api()
    outs = []
    for group in groups:
        runs = api.runs(f'alasdairtran/{dataset}', {
            'config.wandb.group': group,
            'state': 'finished'
        })
        losses = [run.summary['test_loss'] for run in runs]
        if len(losses) != 3:
            print(f'fail {group}, {len(losses)}')
        outs.append(losses)
    return np.array(outs) * 100


def plot_line(xs, losses, ax, axis=1, **kwargs):
    means = losses.mean(axis)
    e_lower = means - losses.min(axis)
    e_upper = losses.max(axis) - means
    yerr = np.array([e_lower, e_upper])
    return ax.errorbar(xs[:len(means)], means, yerr=yerr, **kwargs)


def plot_performance_vs_layer(ax):
    layers_1 = [4, 8, 12, 16, 20]
    layers_2 = [4, 8, 12, 16, 20, 24]
    xs = [4, 8, 12, 16, 20, 24]
    dataset = 'navier-stokes-4'
    lines = []

    groups = [f'ablation/teaching_forcing/{i}_layers' for i in layers_1]
    losses = get_test_losses(dataset, groups)
    container = plot_line(xs, losses, ax, color=pal[1])
    lines.append(container.lines[0])

    groups = [f'zongyi/{i}_layers' for i in layers_1]
    losses = get_test_losses(dataset, groups)
    container = plot_line(xs, losses, ax, color=pal[0], linestyle='--')
    lines.append(container.lines[0])

    groups = [f'ablation/zongyi_markov/{i}_layers' for i in layers_1]
    losses = get_test_losses(dataset, groups)
    container = plot_line(xs, losses, ax, color=pal[2], linestyle=':')
    lines.append(container.lines[0])

    groups = [f'ablation/no_factorization/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    container = plot_line(xs, losses, ax, color=pal[9], linestyle='-')
    lines.append(container.lines[0])

    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Normalized MSE (%)')

    return lines


def plot_ablation(ax):
    layers_2 = [4, 8, 12, 16, 20, 24]
    xs = [4, 8, 12, 16, 20, 24]
    dataset = 'navier-stokes-4'
    lines = []

    groups = [f'ablation/no_factorization/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    container = plot_line(xs, losses, ax, color=pal[9], linestyle='-')
    lines.append(container.lines[0])

    groups = [f'ablation/no_sharing/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    container = plot_line(xs, losses, ax, color=pal[8], linestyle='-.')
    lines.append(container.lines[0])

    groups = [f'markov/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    container = plot_line(xs, losses, ax, color=pal[3])
    lines.append(container.lines[0])

    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Number of Layers')

    return lines


def get_step_losses(dataset, group):
    api = wandb.Api()
    runs = api.runs(f'alasdairtran/{dataset}', {
        'config.wandb.group': group,
        'state': 'finished'
    })
    run_losses = []
    for run in runs:
        losses = []
        for i in range(10):
            losses.append(run.summary[f'test_loss_{i}'])
        run_losses.append(losses)

    return np.array(run_losses) * 100


def plot_step_loss_curves(ax):
    xs = list(range(1, 11))
    dataset = 'navier-stokes-4'

    losses = get_step_losses(dataset, 'zongyi/4_layers')
    plot_line(xs, losses, ax, axis=0, color=pal[0], linestyle='--')

    losses = get_step_losses(dataset, 'ablation/zongyi_markov/4_layers')
    plot_line(xs, losses, ax, axis=0, color=pal[2], linestyle=':')

    losses = get_step_losses(dataset, 'markov/4_layers')
    plot_line(xs, losses, ax, axis=0, color=pal[3])

    ax.set_xticks(xs)
    ax.set_xlabel('Inference Step')


def get_paramter_count(dataset, groups):
    api = wandb.Api()

    param_counts = []
    for group in groups:
        runs = api.runs(f'alasdairtran/{dataset}', {
            'config.wandb.group': group,
            'config.trial': 0,
            'state': 'finished',
        })
        count = runs[0].summary['n_params']
        param_counts.append(count)

    return np.array(param_counts)


def plot_parameters(ax):
    dataset = 'navier-stokes-4'
    xs = [4, 8, 12, 16, 20, 24]
    lines = []

    groups = [f'ablation/no_factorization/{i}_layers' for i in xs]
    counts = get_paramter_count(dataset, groups)
    line = ax.plot(xs, counts, color=pal[9], linestyle='-')
    lines.append(line[0])

    groups = [f'ablation/no_sharing/{i}_layers' for i in xs]
    counts = get_paramter_count(dataset, groups)
    line = ax.plot(xs, counts, color=pal[8], linestyle='-.')
    lines.append(line[0])

    groups = [f'markov/{i}_layers' for i in xs]
    counts = get_paramter_count(dataset, groups)
    line = ax.plot(xs, counts, color=pal[3], linestyle='-')
    lines.append(line[0])

    ax.set_yscale('log')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Parameter Count')
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])

    return lines


def get_inference_times(dataset, groups):
    api = wandb.Api()
    losses, times = [], []
    for group in groups:
        runs = api.runs(f'alasdairtran/{dataset}', {
            'config.wandb.group': group,
            'state': 'finished'
        })
        assert len(runs) == 3
        losses.append([run.summary['test_loss'] for run in runs])
        times.append([run.summary['inference_time'] for run in runs])

    return 100 * np.array(losses), np.array(times)


def plot_xy_line(xs, ys, ax, axis=1, **kwargs):
    y_means = ys.mean(axis)
    e_lower = y_means - ys.min(axis)
    e_upper = ys.max(axis) - y_means
    yerr = np.array([e_lower, e_upper])

    x_means = xs.mean(axis)
    e_lower = x_means - xs.min(axis)
    e_upper = xs.max(axis) - x_means
    xerr = np.array([e_lower, e_upper])

    return ax.errorbar(x_means, y_means, xerr=xerr, yerr=yerr, **kwargs)


def plot_pde_inference_performance_tradeoff(ax):
    dataset = 'navier-stokes-4'
    layers_1 = [4, 8, 12, 16, 20]
    layers_2 = [4, 8, 12, 16, 20, 24]
    lines = []

    groups = [f'markov/{i}_layers' for i in layers_2]
    losses, times = get_inference_times(dataset, groups)
    container = plot_xy_line(losses, times, ax, color=pal[3], linestyle='-')
    lines.append(container.lines[0])

    groups = [f'ablation/no_factorization/{i}_layers' for i in layers_2]
    losses, times = get_inference_times(dataset, groups)
    container = plot_xy_line(losses, times, ax, color=pal[9], linestyle='-')
    lines.append(container.lines[0])

    groups = [f'zongyi/{i}_layers' for i in layers_1]
    losses, times = get_inference_times(dataset, groups)
    container = plot_xy_line(losses, times, ax, color=pal[0], linestyle='--')
    lines.append(container.lines[0])

    ax.scatter([0], [244], color=pal[4])
    ax.set_xlabel('Normalized MSE (%)')
    ax.set_ylabel('Inference Time (s)')
    ax.set_yscale('log')
    ax.set_xticks([0, 5, 10, 15, 20])

    return lines


def plot_pde_training_performance_tradeoff(ax):
    dataset = 'navier-stokes-4'
    layers_1 = [4, 8, 12, 16, 20]
    layers_2 = [4, 8, 12, 16, 20, 24]
    lines = []

    groups = [f'markov/{i}_layers' for i in layers_2]
    losses, times = get_training_times(dataset, groups)
    container = plot_xy_line(losses, times, ax, color=pal[3], linestyle='-')
    lines.append(container.lines[0])

    groups = [f'ablation/no_factorization/{i}_layers' for i in layers_2]
    losses, times = get_training_times(dataset, groups)
    container = plot_xy_line(losses, times, ax, color=pal[9], linestyle='-')
    lines.append(container.lines[0])

    groups = [f'zongyi/{i}_layers' for i in layers_1]
    losses, times = get_training_times(dataset, groups)
    container = plot_xy_line(losses, times, ax, color=pal[0], linestyle='--')
    lines.append(container.lines[0])

    ax.set_xlabel('Normalized MSE (%)')
    ax.set_ylabel('Training Time (h)')
    ax.set_xticks([0, 5, 10, 15, 20])

    return lines


def get_training_times(dataset, groups):
    api = wandb.Api()
    losses, times = [], []
    for group in groups:
        runs = api.runs(f'alasdairtran/{dataset}', {
            'config.wandb.group': group,
            'state': 'finished'
        })
        assert len(runs) == 3
        losses.append([run.summary['test_loss'] for run in runs])
        times.append([run.summary['_runtime'] for run in runs])

    return 100 * np.array(losses), np.array(times) / 3600


def plot_scalable():
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)

    layers_1 = [4, 8, 12, 16, 20]
    layers_2 = [4, 8, 12, 16, 20, 24]
    xs = [4, 8, 12, 16, 20, 24]
    dataset = 'navier-stokes-4'
    lines = []

    groups = [f'zongyi/{i}_layers' for i in layers_1]
    losses = get_test_losses(dataset, groups)
    container = plot_line(xs, losses, ax, color=pal[0], linestyle='--')
    lines.append(container.lines[0])

    groups = [f'markov/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    container = plot_line(xs, losses, ax, color=pal[3])
    lines.append(container.lines[0])

    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Normalized MSE (%)')

    labels = ['FNO [Li et al., 2021a]',
              'F-FNO (our full model)']
    ax.legend(lines, labels)

    fig.tight_layout()
    fig.savefig('figures/scalable.png',
                bbox_inches='tight',
                dpi=300)


def plot_10_steps():
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)

    xs = list(range(1, 11))
    dataset = 'navier-stokes-4'
    lines = []

    losses = get_step_losses(dataset, 'zongyi/4_layers')
    container = plot_line(xs, losses, ax, axis=0, color=pal[0], linestyle='--')
    lines.append(container.lines[0])

    losses = get_step_losses(dataset, 'markov/4_layers')
    container = plot_line(xs, losses, ax, axis=0, color=pal[3])
    lines.append(container.lines[0])

    ax.set_xticks(xs)
    ax.set_xlabel('Inference Step')
    ax.set_ylabel('Normalized MSE (%)')
    labels = ['FNO [Li et al., 2021a]',
              'F-FNO (our full model)']
    ax.legend(lines, labels)

    fig.tight_layout()
    fig.savefig('figures/10_steps.png',
                bbox_inches='tight',
                dpi=300)


def plot_poster_pde_inference():
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)

    dataset = 'navier-stokes-4'
    layers_1 = [4, 8, 12, 16, 20]
    layers_2 = [4, 8, 12, 16, 20, 24]
    lines = []

    groups = [f'zongyi/{i}_layers' for i in layers_1]
    losses, times = get_inference_times(dataset, groups)
    container = plot_xy_line(losses, times, ax, color=pal[0], linestyle='--')
    lines.append(container.lines[0])

    groups = [f'markov/{i}_layers' for i in layers_2]
    losses, times = get_inference_times(dataset, groups)
    container = plot_xy_line(losses, times, ax, color=pal[3], linestyle='-')
    lines.append(container.lines[0])

    ax.scatter([0], [244], color=pal[4])
    ax.set_xlabel('Normalized MSE (%)')
    ax.set_ylabel('Inference Time (s)')
    ax.set_yscale('log')
    ax.set_xticks([0, 5, 10, 15, 20])

    sim_line = Line2D(range(1), range(1), color="white",
                      marker='o', markerfacecolor=pal[4])
    lines = [sim_line] + lines
    labels = ['Crank-Nicolson method',
              'FNO [Li et al., 2021a]',
              'F-FNO (our full model)']
    ax.legend(lines, labels)

    fig.tight_layout()
    fig.savefig('figures/inference.png',
                bbox_inches='tight',
                dpi=300)


if __name__ == "__main__":
    app()
