from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from myterial import grey_dark

from fcutils.plot.elements import plot_mean_and_error

from analysis.ephys.utils import get_frate_per_bin


def raster_histo(ax, X, unit, bins: np.ndarray, n_events: int):
    """
        Makes a histogram visualization to put on top of araster plot
        given a vector of events X
    """
    # get counts per bin
    counts, _ = np.histogram(X, bins=bins)

    # get average count across events
    avg_counts = counts / n_events

    # histogram plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="30%", pad=0.05)

    # cax.hist(
    #     S, bins=np.arange(0, 260 + ds, step=ds), color=unit.color, alpha=1
    # )
    cax.bar(
        bins[0:-1] + bins[1] - bins[0],
        avg_counts,
        width=bins[1] - bins[0],
        color=unit.color,
        alpha=1,
    )
    return cax


def bouts_raster(ax, unit, bouts, tracking, ds=5):
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
    bins = np.arange(0, 260 + ds, step=ds)

    # get avg number of spikes per bin
    counts, _ = np.histogram(S, bins=bins)
    avg_counts = counts / n

    # get the avg number of frames per bin
    gcoord_values = (
        np.hstack(
            [
                tracking.global_coord[b.start_frame : b.end_frame]
                for i, b in bouts.iterrows()
            ]
        )
        * 260
    )
    frames_count, _ = np.histogram(gcoord_values, bins=bins)
    frames_avg = frames_count / n

    # now get the avg number of spikes divided by number of frames
    frate = avg_counts / frames_avg

    # make plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="30%", pad=0.05)
    cax.bar(
        bins[:-1] + (bins[1] - bins[0]) / 2,
        frate,
        width=bins[1] - bins[0],
        color=unit.color,
        alpha=1,
    )

    # style axes
    ax.set(
        yticks=np.arange(0, 1, 10 / n),
        yticklabels=(np.arange(0, 1, 10 / n) * n).astype(int),
        xlim=[0, 260],
        ylabel="trial",
    )
    cax.set(ylabel="Firing rate", xticks=[], xlim=[0, 260])
    # tax.set(
    #     ylabel="Speed (cm/s)",
    #     xticks=[],
    #     xlim=[0, 260],
    #     xlabel="track progression (cm)",
    # )


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
    cax = raster_histo(
        ax,
        perievent_spikes,
        unit,
        np.arange(-t_before, t_after + dt, step=dt),
        n,
    )

    # style axes
    ax.set(
        xlabel="time (s)",
        ylabel="trial",
        yticks=np.arange(0, 1, 10 / n),
        yticklabels=(np.arange(0, 1, 10 / n) * n).astype(int),
        xlim=[-t_before, t_after],
    )
    cax.set(ylabel="Spike counts", xticks=[], xlim=[-t_before, t_after])


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


def plot_tuning_curves(ax, tuning_curves: dict, color: str, xlabel: str = ""):
    """
        Given a dictionary of tuning curves where we have
        as keys the values of the independent values and as 
        values a list of "tuning curve values" generated by
        repeated random samples (see `get_tuning_curves`).
    """
    x = np.array(list(tuning_curves.keys()))
    y = np.array([np.nanmean(v) for v in tuning_curves.values()])
    e = np.array([np.nanstd(v) for v in tuning_curves.values()])

    x = x[y > 0]
    e = e[y > 0]
    y = y[y > 0]
    plot_mean_and_error(y, e, ax, color=color, x=x)

    # for k, v in tuning_curves.items():
    #     x = np.ones_like(v) * k
    #     ax.scatter(x, v, s=4, color=color, alpha=0.3)

    ax.set(ylabel="Firing rate (Hz)", xlabel=xlabel)
