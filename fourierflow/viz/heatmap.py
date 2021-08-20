import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MidpointNormalize(mpl.colors.Normalize):
    """Define zero as midpoint of colour map.

    See https://stackoverflow.com/a/50003503/3790116.
    """

    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(
            0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(
            1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [
            normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def log_navier_stokes_heatmap(expt, tensor, name):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    vals = tensor.cpu().numpy()
    vmax = vals.max()
    vmin = -1 if 'layer' in name else -3
    vmax = 1 if 'layer' in name else 3
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    cmap = plt.get_cmap('RdBu')
    im = ax.imshow(vals, interpolation='bilinear', norm=norm, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.tight_layout()
    expt.log({f'{name}': wandb.Image(fig)})
    plt.close('all')
