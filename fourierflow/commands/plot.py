import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import seaborn as sns
import wandb
import xarray as xr
from jax_cfd.data.evaluation import compute_summary_dataset
from matplotlib.lines import Line2D
from typer import Typer

from fourierflow.utils import calculate_time_until, grid_correlation
from fourierflow.viz.heatmap import MidpointNormalize

pal = sns.color_palette()
app = Typer()


@app.command()
def resolution():
    fig = plt.figure(figsize=(8, 3))

    ax = plt.subplot(1, 2, 1)
    lines_1 = plot_correlation_over_time(ax)

    ax = plt.subplot(1, 2, 2)
    lines_2 = plot_energy_spectrum(ax)

    # lines = lines_1

    # labels = ['Adams-Bashforth numerical simulator',
    #           'F-FNO (our full model)']

    # lgd = fig.legend(handles=lines,
    #                  labels=labels,
    #                  loc="center",
    #                  borderaxespad=0.1,
    #                  bbox_to_anchor=[1.2, 0.55])

    fig.tight_layout()
    fig.savefig('figures/superresolution.pdf',
                # bbox_extra_artists=(lgd,),
                bbox_inches='tight')


@app.command()
def correlation():
    fig = plt.figure(figsize=(8, 3))

    ax = plt.subplot(1, 2, 1)
    lines_1 = plot_correlation_vs_time_of_different_grid_sizes(ax)

    ax = plt.subplot(1, 2, 2)
    lines_2 = plot_varying_step_size(ax)

    lines = lines_1

    labels = ['DNS (Carpenter-Kennedy)',
              'F-FNO (our full model)']

    lgd = fig.legend(handles=lines,
                     labels=labels,
                     loc="center",
                     borderaxespad=0.1,
                     bbox_to_anchor=[1.2, 0.55])

    fig.tight_layout()
    fig.savefig('figures/correlation.pdf',
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')


def plot_correlation_over_time(ax):
    groups = [
        'ffno/superresolution/x32_x64/32',
        'ffno/superresolution/x32_x64/64',
        'ffno/superresolution/x32_x64/128',
        'ffno/superresolution/x32_x64/256',
    ]
    api = wandb.Api()
    dataset = 'kolmogorov_re_1000'

    corrs = []
    times = []
    lines = []
    for group in groups:
        runs = api.runs(f'alasdairtran/{dataset}', {
            'config.wandb.group': group,
            'state': 'finished',
        })
        assert len(runs) == 1

        name = f'{dataset}/run-{runs[0].id}-test_correlations:latest'
        artifact = api.artifact(name)
        table = artifact.get('test_correlations')
        time = table.get_column('time')
        corr = table.get_column('corr')
        corrs.append(np.array(corr))
        times.append(np.array(time))

        line, = ax.plot(time, corr)
        lines.append(line)

    labels = ['32x32', '64x64', '128x128', '256x256']
    ax.legend(lines, labels)
    ax.set_xlabel('Simulation time')
    ax.set_ylabel('Vorticity correlation')


def plot_energy_spectrum(ax):
    sizes = [64, 128, 256, 512, 1024]
    models = {}
    path = f'../experiments/kolmogorov/re_1000/ffno/predictions/64/preds.nc'
    models['ffno_64'] = xr.open_dataset(path)

    path = f'../experiments/kolmogorov/re_1000/ffno/predictions/128/preds.nc'
    models['ffno_128'] = xr.open_dataset(path)

    for size in sizes:
        path = f'../data/kolmogorov/re_1000/baselines/{size}_64.nc'
        models[f'baseline_{size}'] = xr.open_dataset(
            path).isel(time=slice(19, None, 20))

    path = f'../data/kolmogorov/re_1000/trajectories/test_64.nc'
    models['baseline_2048'] = xr.open_dataset(
        path).isel(time=slice(19, None, 20))

    for k, model in models.items():
        model.attrs['ndim'] = 2
        models[k] = model.rename({'vx': 'u', 'vy': 'v'})

    summary = xr.concat([
        compute_summary_dataset(ds, models['baseline_2048'])
        for ds in models.values()
    ], dim='model')
    summary.coords['model'] = list(models.keys())

    # starts from time=12
    spectrum = summary.energy_spectrum_mean.tail(
        time=80).mean('time').compute()

    baseline_palette = sns.color_palette('YlGnBu', n_colors=7)[0:]
    models_color = sns.xkcd_palette(['burnt orange'])
    palette = baseline_palette + models_color

    for color, model in zip(palette, summary['model'].data):
        if 'baseline_64' in model:
            continue
        style = '-' if 'baseline' in model else '--'
        (spectrum.k ** 5 * spectrum).sel(model=model).plot.line(
            color=color, linestyle=style, label=model, linewidth=3, ax=ax)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('')
    ax.set_xlim(3.6, None)
    ax.set_ylim(1.3e9, None)
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Scaled engery spectrum')


def plot_test_losses_over_time(ax):
    groups = [
        'ffno/superresolution/32',
        'ffno/superresolution/64',
        'ffno/superresolution/128',
        'ffno/superresolution/256',
    ]
    api = wandb.Api()
    dataset = 'kolmogorov_re_1000'

    corrs = []
    times = []
    for group in groups:
        runs = api.runs(f'alasdairtran/{dataset}', {
            'config.wandb.group': group,
            'state': 'finished',
        })
        assert len(runs) == 1

        name = f'{dataset}/run-{runs[0].id}-test_losses:latest'
        artifact = api.artifact(name)
        table = artifact.get('test_losses')
        time = table.get_column('time')
        loss = table.get_column('loss')
        corrs.append(np.array(loss))
        times.append(np.array(time))

        ax.plot(time, loss)
        ax.set_xlabel('Simulation time')
        ax.set_ylabel('Normalized MSE (%)')


def plot_correlation_vs_time_of_different_grid_sizes(ax):
    sizes = [32, 64, 128, 256, 512, 1024]
    simulations = {}
    for size in sizes:
        path = f'../data/kolmogorov/re_1000/baselines/{size}_32.nc'
        simulations[size] = xr.open_dataset(path)
    path = '../data/kolmogorov/re_1000/trajectories/test_32.nc'
    simulations[2048] = xr.open_dataset(path)

    combined = xr.concat(simulations.values(), dim='size')
    combined.coords['size'] = sizes + [2048]

    # Even the best model diverges from ground truth by time 10. Thus we
    # only look at the first 10 simulation steps to save computation time.
    w = combined.vorticity.sel(time=slice(10))
    rho = grid_correlation(w, w.sel(size=2048)).compute()

    lines = []

    times_until = calculate_time_until(rho.isel(sample=slice(0, 4)))
    duration = combined.elapsed.mean(dim='sample') / combined.time.max()
    lines.append(ax.errorbar(
        duration[:-1], times_until[:-1], color=pal[4], marker='x'))
    print(duration)
    ax.set_xlabel('Runtime per time unit (s)')
    ax.set_ylabel('Time until correlation < 95%')
    ax.set_xlim(1e-2, 1e1)
    ax.set_xscale('log')
    # array([0.448799, 1.248222, 2.440344, 3.744666, 5.048988, 6.40941 , 0.      ])
    # Compared to original paper:
    # array([1.711046, 2.973293, 4.15139 , 5.497787, 7.208833, 0.      ])

    grids = [32, 64, 128, 256, 512, 1024, 2048]
    for i, s in enumerate(grids):
        xy = (duration[i], times_until[i])
        xytext = (xy[0] * 1.1, xy[1] - 0.3)
        ax.annotate(f'{s}', xy, xytext)

    api = wandb.Api()
    dataset = 'kolmogorov_re_1000'

    groups = ['ffno/step_sizes/20',
              'ffno/grid_sizes/modes_32/with_velocity/128', ]
    #   'ffno/grid_sizes/modes_64/longer/256']
    times = []
    untils = []

    for group in groups:
        runs = api.runs(f'alasdairtran/{dataset}', {
            'config.wandb.group': group,
            'state': 'finished',
        })
        assert len(runs) == 1
        time = [run.summary['inference_time'] for run in runs]
        times.append(np.array(time).mean())

        until = [run.summary['test_reduced_time_until'] for run in runs]
        untils.append(np.array(until).mean())

    lines.append(ax.errorbar(times, untils, color=pal[3], marker='o'))
    print(times)

    grids = [64, 128]
    for i, s in enumerate(grids):
        xy = (times[i], untils[i])
        xytext = (xy[0] * 1.1, xy[1] - 0.3)
        ax.annotate(f'{s}', xy, xytext)

    return lines


def plot_varying_step_size(ax):
    api = wandb.Api()
    dataset = 'kolmogorov_re_1000'
    sizes = [0.25, 0.5, 1, 2, 5, 10, 20, 40, 80]
    step_sizes = [
        0.0035062418008814655,
        0.007012483601762931,
        0.014024967203525862,
        0.028049934407051724,
        0.07012483601762931,
        0.14024967203525862,
        0.28049934407051724,
        0.5609986881410345,
        1.121997376282069,
    ]
    untils = []
    for size in sizes:
        runs = api.runs(f'alasdairtran/{dataset}', {
            'config.wandb.group': f'ffno/step_sizes/{size}',
            'state': 'finished'
        })
        assert len(runs) == 1
        until = [run.summary['valid_time_until'] for run in runs]
        untils.append(np.array(until).mean())

    lines = []
    line = ax.errorbar(step_sizes, untils, color=pal[3], marker='o')
    lines.append(line)

    # Line for numerical solver
    sims = {}
    sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    for size in sizes:
        path = f'../data/kolmogorov/re_1000/time_steps/x{size}_64.nc'
        sims[size] = xr.open_dataset(path, engine='h5netcdf')
    gt_path = '../data/kolmogorov/re_1000/trajectories/valid_64.nc'
    sims[999] = xr.open_dataset(
        gt_path, engine='h5netcdf').isel(time=slice(1, None, 2))  # key is arbitrary

    combined = xr.concat(sims.values(), dim='size')
    combined.coords['size'] = sizes + [999]

    # Even the best model diverges from ground truth by time 10. Thus we
    # only look at the first 10 simulation steps to save computation time.
    w = combined.vorticity.sel(time=slice(10))
    rho = grid_correlation(w, w.sel(size=999)).compute()
    untils = calculate_time_until(rho.isel(sample=slice(0, 4)))[:-1]
    step_sizes = [
        0.028049934407051724,
        0.014024967203525862,
        0.007012483601762931,
        0.0035062418008814655,
        0.0017531209004407328,
        0.0008765604502203664,
        0.0004382802251101832,
        0.0002191401125550916,
    ]
    line = ax.errorbar(step_sizes, untils, color=pal[4], marker='x')
    lines.append(line)
    ax.set_xscale('log')

    ax.set_xlabel('Step size')
    ax.set_ylabel('Time until correlation < 95%')

    return lines


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
    labels = ['Crank-Nicolson numerical simulator',
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
    dataset = 'ns_zongyi_4'
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
    dataset = 'ns_zongyi_4'
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
    dataset = 'ns_zongyi_4'
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
    dataset = 'ns_zongyi_4'

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
    dataset = 'ns_zongyi_4'
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
    dataset = 'ns_zongyi_4'
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
    dataset = 'ns_zongyi_4'
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
    dataset = 'ns_zongyi_4'
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
    dataset = 'ns_zongyi_4'
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

    dataset = 'ns_zongyi_4'
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
