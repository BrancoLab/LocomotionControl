from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.signal import medfilt

from myterial import grey_dark, blue_grey

from data.data_utils import convolve_with_gaussian

from analysis.ephys.utils import get_frate_per_bin


def bouts_raster(ax, unit, bouts, tracking, ds=1):
    """
        Plot a unit's spikes aligned to bouts. Unlike time_aligned_raster, this function 
        plots spikes as a function of track progression, not time!
    """

    S, Y = [], []
    n = len(bouts)
    h = 1 / n

    for i, bout in bouts.iterrows():
        # get spikes during bout (based on frames)
        trial_spikes = unit.spikes[
            (unit.spikes >= bout.start_frame) & (unit.spikes < bout.end_frame)
        ]

        spikes_s = tracking.global_coord[trial_spikes] * 260
        S.extend(spikes_s)
        Y.extend(np.zeros_like(trial_spikes) + (i * h))

        # mark the start/end of the bout
        ax.scatter(
            [
                tracking.global_coord[bout.start_frame] * 260,
                tracking.global_coord[bout.end_frame] * 260,
            ],
            [i * h, i * h],
            s=24,
            color="red",
            alpha=1,
            marker="|",
        )

    ax.scatter(S, Y, s=4, color=grey_dark, alpha=1, marker="|")

    # histogram plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="30%", pad=0.05)
    cax.hist(
        S, bins=np.arange(0, 260 + ds, step=ds), color=unit.color, alpha=1
    )

    # plot tracking stuff
    tax = divider.append_axes("bottom", size="20%", pad=0.05)
    speed = convolve_with_gaussian(tracking.speed, 11)
    # avel = convolve_with_gaussian(tracking.thetadot, 11)
    for i, bout in bouts.iterrows():
        tax.scatter(
            medfilt(
                tracking.global_coord[bout.start_frame : bout.end_frame] * 260,
                11,
            ),
            speed[bout.start_frame : bout.end_frame],
            color=blue_grey,
            alpha=1,
            s=2,
        )

    ax.set(
        yticks=np.arange(0, 1, 10 / n),
        yticklabels=(np.arange(0, 1, 10 / n) * n).astype(int),
        xlim=[0, 260],
        ylabel="trial",
    )
    cax.set(ylabel="Spike counts", xticks=[], xlim=[0, 260])
    tax.set(
        ylabel="Speed (cm/s)",
        xticks=[],
        xlim=[0, 260],
        xlabel="track progression (cm)",
    )
    return ax, cax


def time_aligned_raster(ax, unit, timestamps, t_before=1, t_after=1, dt=0.1):
    """
        Plot a unit spikes aligned to timestamps (in seconds).
        it also adds a firing rate visualization
    """
    ax.plot([0, 0], [0, 1], lw=3, color="k", alpha=0.3)
    n = len(timestamps)
    h = 1 / n

    spikes = unit.spikes_ms / 1000
    perievent_spikes = []
    Y = []
    for i, t in enumerate(timestamps):
        trial_spikes = spikes[(spikes > t - t_before) & (spikes < t + t_after)]
        y = np.zeros_like(trial_spikes) + (i * h)

        Y.extend(y)
        perievent_spikes.extend(trial_spikes - t)
    ax.scatter(perievent_spikes, Y, s=4, color=grey_dark, alpha=1, marker="|")

    # add horizontal cax to axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="30%", pad=0.05)
    cax.axvline(0, lw=3, color="k", alpha=0.3, zorder=-1)
    cax.hist(
        perievent_spikes,
        bins=np.arange(-t_before, t_after + dt, step=dt),
        color=unit.color,
        alpha=1,
    )

    ax.set(
        xlabel="time (s)",
        ylabel="trial",
        yticks=np.arange(0, 1, 10 / n),
        yticklabels=(np.arange(0, 1, 10 / n) * n).astype(int),
        xlim=[-t_before, t_after],
    )
    cax.set(ylabel="Spike counts", xticks=[], xlim=[-t_before, t_after])
    return ax, cax


def plot_frate_binned_by_var(
    ax, unit, in_bin: dict, bin_values, xlabel="", color=None
):
    """
        Plot a unit's firing rate binned by a variable (with bins defined over the variable's range)
    """
    color = color or unit.color
    # get firing rate per bin
    in_bin_frate = get_frate_per_bin(unit, in_bin)

    # plot firing rate
    ax.plot(bin_values, in_bin_frate.values(), "-o", color=color, lw=2)

    ax.set(xlabel=xlabel, xticks=bin_values[::2])
