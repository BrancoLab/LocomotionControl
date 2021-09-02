
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

from myterial import blue_dark, salmon_dark, indigo


def plot_signal_and_events(signal:np.ndarray, events:Union[np.ndarray, list], secondary_events:np.ndarray=None, show:bool=False):
    '''
        Plots a signals and marks event times on it
    '''
    f, ax = plt.subplots(figsize=(16, 8))

    ax.plot(signal, color=[.3, .3, .3])
    ax.scatter(events, signal[events], color='salmon', marker='v', zorder=100)

    if secondary_events is not None:
        ax.scatter(secondary_events, signal[secondary_events], color='green', marker='v', zorder=999, alpha=.7)

    if show:
        plt.show()


def plot_recording_triggers(
    bonsai_sync,
    ephys_sync,
    bonsai_sync_onsets,
    bonsai_sync_offsets,
    ephys_sync_onsets,
    ephys_sync_offsets,
    sampling_rate,
    time_scaling_factor,
):
    N = 50
    bonsai_time = (
        np.arange(0, len(bonsai_sync), step=N) - bonsai_sync_onsets[0]
    ) / sampling_rate
    probe_time = (
        (np.arange(0, len(ephys_sync), step=N) - ephys_sync_onsets[0])
        / sampling_rate
        * time_scaling_factor
    )

    # plot sync signals
    f, axes = plt.subplots(
        figsize=(16, 8), ncols=2, gridspec_kw={"width_ratios": [3, 1]}
    )
    axes[0].plot(bonsai_time, bonsai_sync[::N] / 5, lw=0.5, color="b")
    axes[0].plot(probe_time, ephys_sync[::N] * 0.02, lw=0.5, color="salmon")

    # mark trigger onsets
    bad_onsets = np.where(np.diff(bonsai_sync_onsets) != sampling_rate)[0]
    probe_bad_onsets = np.where(
        np.abs(np.diff(ephys_sync_onsets) - sampling_rate) > sampling_rate / 2
    )[0]

    axes[0].scatter(
        (bonsai_sync_onsets[bad_onsets] - bonsai_sync_onsets[0])
        / sampling_rate,
        np.ones_like(bad_onsets) * 1.4,
        marker="v",
        color=indigo,
        alpha=0.75,
        s=75,
        zorder=120,
    )
    axes[0].scatter(
        (bonsai_sync_onsets - bonsai_sync_onsets[0]) / sampling_rate,
        np.ones_like(bonsai_sync_onsets) * 1.2,
        marker="v",
        color=blue_dark,
        zorder=100,
    )

    axes[0].scatter(
        (ephys_sync_onsets - ephys_sync_onsets[0])
        / sampling_rate
        * time_scaling_factor,
        np.ones_like(ephys_sync_onsets) * 1.25,
        marker="v",
        color=salmon_dark,
        zorder=100,
    )
    axes[0].scatter(
        (ephys_sync_onsets[probe_bad_onsets] - ephys_sync_onsets[0])
        / sampling_rate
        * time_scaling_factor,
        np.ones_like(probe_bad_onsets) * 1.45,
        marker="v",
        s=85,
        alpha=0.7,
        color=salmon_dark,
        zorder=100,
    )

    axes[0].set(title="blue: bonsai | salmon: probe", xlabel="time (s)")

    # plot histogram of triggers durations
    axes[1].hist(np.diff(bonsai_sync_onsets), bins=50, color="b", alpha=0.5)
    axes[1].hist(
        np.diff(ephys_sync_onsets), bins=50, color="salmon", alpha=0.5
    )
    axes[1].set(title="blue: bonsai | salmon: probe", xlabel="offset duration")
