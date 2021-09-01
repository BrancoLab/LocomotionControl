import pandas as pd
import numpy as np
from typing import Union

KEYS = (
    "x",
    "y",
    "segment",
    "global_coord",
    "speed",
    "orientation",
    "direction_of_movement",
    "angular_velocity",
)


def bin_x_by_y(data:pd.DataFrame, x:str, y:str, bins:Union[int, np.ndarray]=10):
    '''
        Bins the values in a column X of a dataframe by bins
        specified based on the values of another column Y
    '''
    data['bins'], bins = pd.cut(data[y], bins=bins, retbins=True)
    data = data.loc[data.bins != np.nan]

    bins_centers = np.cumsum(np.diff(bins)) - np.diff(bins)[0]/2

    mu = data.groupby('bins')[x].mean()
    sigma = data.groupby('bins')[x].std()
    return bins_centers, mu, sigma

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