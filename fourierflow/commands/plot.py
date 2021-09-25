import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from typer import Typer

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
    labels = ['FNO proposed by Li et al. [2021a]',
              'FNO with teacher forcing',
              'FNO with Markov assumption',
              'FNO with bags of tricks',
              'Factorized FNO without weight sharing',
              'Factorized FNO (F-FNO)']
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
    plot_parameters(ax)

    ax = plt.subplot(1, 3, 2)
    plot_pde_inference_performance_tradeoff(ax)

    fig.tight_layout()
    fig.savefig('figures/complexity.pdf',
                bbox_inches='tight')


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
    container = plot_line(xs, losses, ax, color=pal[7], linestyle='-')
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
    container = plot_line(xs, losses, ax, color=pal[7], linestyle='-')
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
    xs = list(range(10))
    dataset = 'navier-stokes-4'

    losses = get_step_losses(dataset, 'zongyi/4_layers')
    plot_line(xs, losses, ax, axis=0, color=pal[0], linestyle='--')

    losses = get_step_losses(dataset, 'ablation/zongyi_markov/4_layers')
    plot_line(xs, losses, ax, axis=0, color=pal[2], linestyle=':')

    losses = get_step_losses(dataset, 'markov/4_layers')
    plot_line(xs, losses, ax, axis=0, color=pal[3])

    ax.set_xticks(xs)
    ax.set_xlabel('Inference Step')


def plot_pde_inference_performance_tradeoff(ax):
    # 4 3.254133462905884
    # 8 6.324826240539551
    # 12 9.374979257583618
    # 16 12.499497175216675
    # 20 15.55120301246643
    # 24 18.649063110351562
    # 243.98698592185974
    x = [0.05705, 0.03422, 0.02861, 0.02613, 0.02408, 0.02287, 0]
    y = [3.254133462905884, 6.324826240539551, 9.374979257583618,
         12.499497175216675, 15.55120301246643, 18.649063110351562,
         243.98698592185974]
    x = np.array(x) * 100

    ax.scatter(x, y)
    ax.set_xlabel('Normalized MSE (%)')
    ax.set_ylabel('Inference Time (s)')


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

    groups = [f'ablation/no_factorization/{i}_layers' for i in xs]
    counts = get_paramter_count(dataset, groups)
    ax.plot(xs, counts, color=pal[7], linestyle='-')

    groups = [f'ablation/no_sharing/{i}_layers' for i in xs]
    counts = get_paramter_count(dataset, groups)
    ax.plot(xs, counts, color=pal[8], linestyle='-.')

    groups = [f'markov/{i}_layers' for i in xs]
    counts = get_paramter_count(dataset, groups)
    ax.plot(xs, counts, color=pal[3], linestyle='-')

    ax.set_yscale('log')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Parameter Count')


if __name__ == "__main__":
    app()
