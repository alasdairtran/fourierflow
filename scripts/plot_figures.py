import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

pal = sns.color_palette()


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
    plot_line(xs, losses, ax, color=pal[0])

    groups = [f'ablation/teaching_forcing/{i}_layers' for i in layers_1]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax, color=pal[1])

    groups = [f'ablation/zongyi_markov/{i}_layers' for i in layers_1]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax, color=pal[2])

    groups = [f'markov/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax, color=pal[3])

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
    plot_line(xs, losses, ax, color=pal[7])

    groups = [f'ablation/no_sharing/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax, color=pal[8])

    groups = [f'markov/{i}_layers' for i in layers_2]
    losses = get_test_losses(dataset, groups)
    plot_line(xs, losses, ax, color=pal[3])

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

    fig = plt.figure(figsize=(3.2, 3))
    ax = plt.subplot(1, 1, 1)
    ax.scatter(x, y)
    ax.set_xlabel('Normalized MSE')
    ax.set_ylabel('Inference time')
    ax.set_title('PDE Inference Performance Tradeoff')

    fig.tight_layout()
    fig.savefig('figures/tradeoff.pdf')
