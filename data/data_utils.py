import pandas as pd
import numpy as np
from typing import Union, Tuple, List
from loguru import logger
from scipy import stats
from scipy.signal import medfilt

from fcutils.maths.signals import get_onset_offset

from data.debug_utils import plot_signal_and_events

from kino.locomotion import Locomotion


KEYS = (
    "x",
    "y",
    "segment",
    "global_coord",
    "speed",
    "orientation",
    "direction_of_movement",
    "angular_velocity",
    "spikes",
    "firing_rate",
    "dmov_velocity",
    "acceleration",
    "theta",
    "thetadot",
    "thetadotdot",
)


def merge_locomotion_bouts(bouts: List[Locomotion]) -> Tuple[np.ndarray]:
    """
        It concats scalar quantities across individual bouts
        X -> x pos
        Y -> y pos
        S -> speed
        A -> acceleration
        T -> theta/orientation
        AV -> angular velocity
        AA -> angular acceleration
        LonA -> longitudinal acceleartion
        LatA -> lateral acceleration
    """
    X, Y, S, A, T, AV, AA, LonA, LatA = [], [], [], [], [], [], [], [], []

    for bout in bouts:
        start = np.where(bout.body.speed > 10)[0][0]
        X.append(bout.body.x[start:])
        Y.append(bout.body.y[start:])
        S.append(bout.body.speed[start:])
        A.append(bout.body.acceleration_mag[start:])
        T.append(bout.body.theta[start:])
        AV.append(bout.body.thetadot[start:])
        AA.append(bout.body.thetadotdot[start:])
        LonA.append(bout.body.longitudinal_acceleration[start:])
        LatA.append(bout.body.normal_acceleration[start:])

    return (
        np.hstack(X),
        np.hstack(Y),
        np.hstack(S),
        np.hstack(A),
        np.hstack(T),
        np.hstack(AV),
        np.hstack(AA),
        np.hstack(LonA),
        np.hstack(LatA),
    )


def resample_linear_1d(original: np.ndarray, target_length: int) -> np.ndarray:
    """
        Similar to scipy resample but with no aberration, see:
            https://stackoverflow.com/questions/20322079/downsample-a-1d-numpy-array
    """
    original = np.array(original, dtype=np.float)
    index_arr = np.linspace(
        0, len(original) - 1, num=target_length, dtype=np.float
    )
    index_floor = np.array(index_arr, dtype=np.int)  # Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor  # Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0 - index_rem) + val2 * index_rem
    assert len(interp) == target_length
    return interp


def register_in_time(
    arrays: List[np.ndarray], n_samples: int = None
) -> List[np.ndarray]:
    """
        Given a list of 1d numpy arrays of different length,
        this function returns an array of shape (n_samples, n_trials) so
        that each trial has the same number of samples and can thus be averaged
        nicely
    """
    n_samples = n_samples or np.min([len(x) for x in arrays])
    return [resample_linear_1d(x, n_samples) for x in arrays]


def mean_and_std(arrays: List[np.ndarray]) -> Tuple[np.ndarray]:
    """
        Given a list of 1D arrays of same length, returns the mean
        and std arrays
    """
    X = np.vstack(arrays)
    return np.mean(X, 0), np.mean(X, 1)


def remove_outlier_values(
    data: np.ndarray,
    threshold: Union[Tuple[float, float], float],
    errors_calculation_array: np.ndarray = None,
) -> np.ndarray:
    """
        Removes extreme values form an array by setting them to nan and interpolating what's left
    """
    dtype = data.dtype

    # find where extreme values are
    if errors_calculation_array is None:
        errors_calculation_array = data.copy()

    if isinstance(threshold, tuple):
        errors = np.where(
            (errors_calculation_array > threshold[0])
            & (errors_calculation_array < threshold[1])
        )[0]
    else:
        errors = np.where(errors_calculation_array > threshold)[0]

    data[errors - 1] = np.nan
    data[errors] = np.nan
    data = interpolate_nans(data=data)["data"]
    return np.array(list(data.values())).astype(dtype)


def convolve_with_gaussian(
    data: np.ndarray, kernel_width: int = 21
) -> np.ndarray:
    """
        Convolves a 1D array with a gaussian kernel of given width
    """
    # create kernel and normalize area under curve
    norm = stats.norm(0, kernel_width)
    X = np.linspace(norm.ppf(0.0001), norm.ppf(0.9999), kernel_width)

    _kernel = norm.pdf(X)
    kernel = _kernel / np.sum(_kernel)

    padded = np.pad(data, 2 * kernel_width, mode="edge")
    return np.convolve(padded, kernel, mode="same")[
        2 * kernel_width : -2 * kernel_width
    ]


def pd_series_to_df(series: pd.Series) -> pd.DataFrame:
    """
        Converts a series to a dataframe
    """
    keys = [k for k in KEYS if k in list(series.index)]
    return pd.DataFrame({k: series[k] for k in keys})


def get_event_times(
    data: np.ndarray,
    kernel_size: int = 71,
    skip_first: int = 20 * 60,
    th: float = 0.1,
    abs_val: bool = False,
    shift: int = 0,
    debug: bool = False,
) -> Tuple[list, list]:
    """
        Given a 1D time serires it gets all the times there's a new 'stimulus' (signal going > threshold).
    """
    original = data.copy()
    if abs_val:
        data = np.abs(data)
    if kernel_size is not None:
        data = medfilt(data, kernel_size=kernel_size)
    onsets, offsets = get_onset_offset(data, th=th)
    onsets = [on for on in onsets if on > offsets[0]]

    # skip onsets that occurred soon after the session start
    onsets = [on - shift for on in onsets if on > skip_first]
    offsets = [off - shift for off in offsets if off > onsets[0]]

    # check
    if len(onsets) != len(offsets):
        raise ValueError("Error while getting event times")

    for on, off in zip(onsets, offsets):
        if on > off:
            raise ValueError("Onsets cant be after offset")

    if debug:
        logger.debug(f"Found {len(onsets)} event times")
        # visualize event times
        plot_signal_and_events(
            data, onsets, offsets, second_signal=original, show=True
        )
    return onsets, offsets


def bin_x_by_y(
    data: Union[pd.DataFrame, pd.Series],
    x: str,
    y: str,
    bins: Union[int, np.ndarray] = 10,
    min_count: int = 0,
) -> Tuple[np.ndarray, float, float, int]:
    """
        Bins the values in a column X of a dataframe by bins
        specified based on the values of another column Y
    """
    if isinstance(data, pd.Series):
        data = pd_series_to_df(data)

    # get bins
    data["bins"], bins = pd.cut(data[y], bins=bins, retbins=True)
    data = data.loc[data.bins != np.nan]
    bins_centers = (
        bins[0] + np.cumsum(np.diff(bins)) - abs(np.diff(bins)[0] / 2)
    )

    # get values
    mu = data.groupby("bins")[x].mean()
    sigma = data.groupby("bins")[x].std()
    counts = data.groupby("bins")[x].count()

    # remove bins with values too low
    mu[counts < min_count] = np.nan
    sigma[counts < min_count] = np.nan
    counts[counts < min_count] = np.nan
    return bins_centers, mu, sigma, counts


def interpolate_nans(**entries) -> dict:
    return (
        pd.DataFrame(entries)
        .interpolate(in_place=True, axis=0, limit_direction="both")
        .to_dict()
    )


def select_by_indices(tracking: dict, selected_indices: np.ndarray) -> dict:
    """
        Given a dictionary of tracking data it select data at given timestamps/indices
    """
    tracking = tracking.copy()
    for key in KEYS:
        if key in tracking.keys():
            tracking[key] = tracking[key][selected_indices]
    return tracking


def downsample_tracking_data(tracking: dict, factor: int = 10) -> None:
    """
        Downsamples tracking data to speed plots and stuff
    """
    for key in KEYS:
        if key in tracking.keys():
            tracking[key] = tracking[key][::10]


def bin_tracking_data_by_arena_position(tracking: dict) -> pd.DataFrame:
    """
        Givena  dictionary with tracking data from the hairpin, 
        including linearized tracking, it bins the tracking data by
        the arena segment the mouse is in 
    """
    tracking_df = pd.DataFrame(
        dict(
            x=tracking["x"],
            y=tracking["y"],
            speed=tracking["speed"],
            orientation=tracking["orientation"],
            direction_of_movement=tracking["direction_of_movement"],
            segment=tracking["segment"],
            global_coord=tracking["global_coord"],
        )
    )

    # get mean and std of each value
    means = tracking_df.groupby("segment").mean()
    stds = tracking_df.groupby("segment").std()
    stds.columns = stds.columns.values + "_std"

    return pd.concat([means, stds], axis=1)
