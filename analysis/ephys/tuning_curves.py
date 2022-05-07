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
    for i in range(len(bins)-1):
        in_bin[i] = np.where((x > bins[i]) & (x <= bins[i + 1]))[0]
    return in_bin


def upsample_farmes_to_ms(var):
    """
        Interpolates the values of a variable expressed in frams (60 fps)
        to values expressed in milliseconds.
    """
    time_frames = np.arange(0, len(var)+1) * 1000/60  # n ms at each frame
    f = interpolate.interp1d(time_frames, var)
    interpolated_variable_values = f(np.arange(0, time_frames[1], 1))
    return interpolated_variable_values


def get_tuning_curves(spike_times: np.ndarray, variable_values: np.ndarray, bins:np.ndarray, n_repeats:int = 10, sample_frac:float=.4) -> dict:
    """
        Get tuning curves of firing rate wrt variables.
        Spike times and variable values are both in milliseconds

        This function gets many tuning curves by repeatedly performing random samples from the data.

        Returns a dictionary of n_repeats values at each bin in bins with the firing rate for a random sample of the data.
    """

    # get max 1 spike per 1ms bin
    spike_times = np.unique(spike_times.astype(int))   # in ms

    # get variable values at spike times
    spike_variable_values = variable_values[spike_times]

    # get tuning curves
    tuning_curves = {v:[] for v in bins}
    for i in range(n_repeats):
        # get random sample
        sample_indices = np.random.choice(len(spike_times), int(sample_frac * len(spike_times)), replace=False)
        sample_spike_times = spike_times[sample_indices]
        sample_spike_variable_values = spike_variable_values[sample_indices]

        in_bin_indices = get_samples_in_bin(sample_spike_variable_values, bins)
        for i, bin_frames in in_bin_indices.items():
            n_spikes = np.sum(np.isin(sample_spike_times, bin_frames))
            n_ms = len(bin_frames)
            tuning_curves[bins[i]].append(n_spikes / n_ms if n_ms > 0 else np.nan)

    return tuning_curves