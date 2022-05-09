import numpy as np
from scipy import interpolate

"""
Code to compute tuning curves of firing rate wrt variables
"""


def get_samples_in_bin(x: np.ndarray, bins: np.ndarray) -> dict:
    """
        Get which samples from a vector of values are in which bin
    """
    in_bin = dict()
    for i in range(len(bins) - 1):
        in_bin[i] = np.where((x > bins[i]) & (x <= bins[i + 1]))[0]
    return in_bin


def upsample_farmes_to_ms(var):
    """
        Interpolates the values of a variable expressed in frams (60 fps)
        to values expressed in milliseconds.
    """
    time_frames = np.arange(0, len(var)) * 1000 / 60  # n ms at each frame
    f = interpolate.interp1d(time_frames, var)
    interpolated_variable_values = f(np.arange(0, time_frames[-1], 1))
    return interpolated_variable_values


def get_tuning_curves(
    spike_times: np.ndarray,
    variable_values: np.ndarray,
    bins: np.ndarray,
    n_frames_sample=10000,
    n_repeats: int = 10,
    sample_frac: float = 0.4,
) -> dict:
    """
        Get tuning curves of firing rate wrt variables.
        Spike times and variable values are both in milliseconds

        Returns a dictionary of n_repeats values at each bin in bins with the firing rate for a random sample of the data.
    """

    # get max 1 spike per 1ms bin
    spike_times = np.unique(spike_times.astype(int))  # in ms

    # get which frames are in which bin
    in_bin_indices = get_samples_in_bin(variable_values, bins)

    # get tuning curves
    tuning_curves = {(v + bins[1] - bins[0]): [] for v in bins[:-1]}
    for i in range(n_repeats):
        # sample n_frames_sample frames from each bin
        sampled_frames = [
            np.random.choice(v, size=n_frames_sample, replace=True)
            if len(v) > n_frames_sample / 3
            else []
            for v in in_bin_indices.values()
        ]

        # get firing rate for each bin
        for i, b in enumerate(tuning_curves.keys()):
            # get spiikes in bin's sampled frames
            if sampled_frames:
                spikes_in_bin = spike_times[
                    np.isin(spike_times, sampled_frames[i])
                ]
                tuning_curves[b].append(
                    len(spikes_in_bin) / n_frames_sample * 1000
                )  # in Hz
            else:
                tuning_curves[b].append(np.nan)

    return tuning_curves
