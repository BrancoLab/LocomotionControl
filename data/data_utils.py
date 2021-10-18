import pandas as pd
import numpy as np
from typing import Union, Tuple
from loguru import logger
from scipy import stats
from scipy.signal import medfilt

from fcutils.maths.signals import get_onset_offset

from data.debug_utils import plot_signal_and_events

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
)

def remove_outlier_values(data:np.ndarray, threshold:Union[Tuple[float, float], float], errors_calculation_array:np.ndarray=None) -> np.ndarray:
    '''
        Removes extreme values form an array by setting them to nan and interpolating what's left
    '''
    dtype = data.dtype

    # find where extreme values are
    if errors_calculation_array is None:
        errors_calculation_array = data.copy()
    
    if isinstance(threshold, tuple):
        errors = np.where((errors_calculation_array > threshold[0])&(errors_calculation_array < threshold[1]))[0]
    else:
        errors = np.where(errors_calculation_array > threshold)[0]
    
    data[errors-1] = np.nan
    data[errors] = np.nan
    data = interpolate_nans(data=data)['data']
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

    _kernnel = norm.pdf(X)
    kernel = _kernnel / np.sum(_kernnel)

    padded = np.pad(data, kernel_width, mode='edge')
    return np.convolve(padded, kernel, mode="same")[kernel_width:-kernel_width]


def get_bouts_tracking_stacked(
    tracking: Union[pd.Series, pd.DataFrame], bouts: pd.DataFrame
) -> pd.DataFrame:
    """
        Creates a dataframe with the tracking data of each bout stacked
    """
    columns = (
        tracking.columns
        if isinstance(tracking, pd.DataFrame)
        else tracking.index
    )
    results = {k: [] for k in KEYS if k in columns}

    for i, bout in bouts.iterrows():
        for key in results.keys():
            results[key].extend(
                list(tracking[key][bout.start_frame : bout.end_frame])
            )
    return pd.DataFrame(results)


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
