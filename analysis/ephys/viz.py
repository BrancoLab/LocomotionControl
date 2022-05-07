import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from myterial import grey_dark

from analysis.ephys.utils import get_frate_per_bin

def time_aligned_raster(ax, unit, timestamps, t_before=1, t_after=1, dt=.1):
    """
        Plot a unit spikes aligned to timestamps (in seconds).
        it also adds a firing rate visualization
    """
    ax.plot([0, 0], [0,1], lw=3, color="k", alpha=.3)
    n = len(timestamps)
    h = 1/n

    spikes = unit.spikes_ms / 1000
    perievent_spikes = []
    X, Y = [], []
    for i,t in enumerate(timestamps):
        trial_spikes = spikes[(spikes > t-t_before) & (spikes < t+t_after)]
        y = np.zeros_like(trial_spikes) + (i * h)

        Y.extend(y)
        perievent_spikes.extend(trial_spikes-t)
    ax.scatter(perievent_spikes, Y, s=4, color=grey_dark, alpha=1, marker=7)

    # add horizontal cax to axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='30%', pad=0.05)
    cax.axvline(0, lw=3, color="k", alpha=.3, zorder=-1)
    cax.hist(perievent_spikes, bins=np.arange(-t_before, t_after+dt, step=dt), color=unit.color, alpha=1)

    ax.set(
        xlabel="time (s)",
        ylabel="trial",
        yticks=np.arange(0, 1, 10/n),
        yticklabels=(np.arange(0, 1, 10/n) * n).astype(int),
        xlim=[-t_before, t_after],
        
    )
    cax.set(ylabel="Spike counts", xticks=[], )
    return ax, cax



def plot_frate_binned_by_var(ax, unit, in_bin:dict, bin_values, xlabel="", color=None):
    """
        Plot a unit's firing rate binned by a variable (with bins defined over the variable's range)
    """
    color = color or unit.color
    # get firing rate per bin
    in_bin_frate = get_frate_per_bin(unit, in_bin)

    # plot firing rate
    ax.plot(bin_values, in_bin_frate.values(), "-o", color=color, lw=2)

    ax.set(xlabel=xlabel, xticks=bin_values[::2])
