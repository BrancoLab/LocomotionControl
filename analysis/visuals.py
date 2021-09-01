import matplotlib.pyplot as plt
from typing import Union
import pandas as pd
import numpy as np
from loguru import logger

from myterial import blue_grey, blue_grey_dark

from data import colors, data_utils



# ----------------------------------- misc ----------------------------------- #
def plot_balls_errors(
    x: np.ndarray, y: np.ndarray, yerr: np.ndarray, ax: plt.axis, colors: list=None
):
    """
        Given a serires of XY values and Y errors it plots a scatter for each XY point and a line
        to mark each Y error
    """
    ax.scatter(x, y, s=150, c=colors, zorder=100, lw=1, ec=[0.3, 0.3, 0.3])
    if colors is None:
        colors = [blue_grey] * len(x)
    elif isinstance(colors, str):
        colors = [colors] * len(x)

    ax.plot(x, y, lw=1, color=blue_grey, zorder=-1)

    for n in range(len(x)):
        ax.plot(
            [x[n], x[n]],
            [y[n] - yerr[n], y[n] + yerr[n]],
            lw=6,
            color=[0.3, 0.3, 0.3],
            zorder=96,
            solid_capstyle="round",
        )
        ax.plot(
            [x[n], x[n]],
            [y[n] - yerr[n], y[n] + yerr[n]],
            lw=4,
            color=colors[n],
            zorder=98,
            solid_capstyle="round",
        )


def plot_bin_x_by_y(data:pd.DataFrame, x:str, y:str, ax:plt.axis, bins:Union[int, np.ndarray]=10, **kwargs):
    '''
        Bin the values of a column X of a dataframe based on the values of
        another column Y and plot as a balls and errors plot
    '''
    # bin
    bins, means, errors = data_utils.bin_x_by_y(data, x, y, bins=bins)

    # plot
    plot_balls_errors(
        bins, means, errors, ax, **kwargs
    )


def plot_aligned(
    x: np.ndarray,
    indices:Union[np.ndarray, list],
    ax:plt.axis,
    mode:str,
    window:int=120,
    **kwargs,
):
    '''
        Given a 1d array and a series of indices it plots the values of 
        the array aligned to the timestamps.
    '''
    # get plotting params
    if mode == 'pre':
        pre_c, pre_lw = kwargs.pop('color', 'salmon'), kwargs.pop('lw', 2)
        aft_c, aft_lw = blue_grey, 1
    else:
        aft_c, aft_lw = kwargs.pop('color', 'salmon'), kwargs.pop('lw', 2)
        pre_c, pre_lw = blue_grey, 1

    pre, aft = int(window/2), int(window/2)
    for idx in indices:
        x_pre = x[idx-pre:idx]
        x_aft = x[idx-1:idx+aft]

        if len(x_pre) != pre or len(x_aft) != aft+1:
            logger.warning(f'Could not plot data aligned to index: {idx}')
            continue

        ax.plot(x_pre, color=pre_c, lw=pre_lw, **kwargs)
        ax.plot(np.arange(aft+1) + aft, x_aft, color=aft_c, lw=aft_lw, **kwargs)
    ax.axvline(pre, lw=2, color=blue_grey_dark, zorder=-1)

def plot_heatmap_2d(
    data:pd.DataFrame,
    key:str,
    ax:plt.axis,
    cmap:str='bwr',
    vmin:int=0,
    vmax:int=100,
    gridsize:int=40,
    mincnt:int = 1,
    **kwargs,
):
    # bin data in 2d
    ax.hexbin(data.x, data.y, data[key], cmap=cmap, gridsize=gridsize, vmin=vmin, vmax=vmax, mincnt=mincnt, **kwargs)


def plot_bouts_x(
    tracking_data:pd.DataFrame,
    bouts:pd.DataFrame,
    ax:plt.axis,
    variable:str,
    color:str=blue_grey,
    **kwargs
):
    for i, bout in bouts.iterrows():
        ax.plot(tracking_data[variable][bout.start_frame:bout.end_frame], color=color, **kwargs)


# ---------------------------------------------------------------------------- #
#                                     EPHSY                                    #
# ---------------------------------------------------------------------------- #
def plot_probe_electrodes(
    rsites: pd.DataFrame, ax: plt.axis, TARGETS: list = [], annotate_every: int=5
):
    x = np.ones(len(rsites)) * 1.025
    x[::2] = 0.925
    x[2::4] = 0.975
    x[1::4] = 1.075

    colors = [
        rs.color
        if rs.brain_region in TARGETS
        else ([0.3, 0.3, 0.3] if rs.color == "k" else blue_grey)
        for i, rs in rsites.iterrows()
    ]
    ax.scatter(
        x,
        rsites.probe_coordinates,
        s=25,
        lw=0.5,
        ec=[0.3, 0.3, 0.3],
        marker="s",
        c=colors,
    )

    for i in range(len(x)):
        if i % annotate_every == 0:
            ax.annotate(
                f"{rsites.site_id.iloc[i]} - {rsites.brain_region.iloc[i]}",
                (0.6, rsites.probe_coordinates.iloc[i]),
                color=colors[i],
            )
    ax.set(xlim=[0.5, 1.25], ylabel="probe coordinates (um)")

# ---------------------------------------------------------------------------- #
#                                   TRACKING                                   #
# ---------------------------------------------------------------------------- #
# -------------------------------- linearized -------------------------------- #
def plot_tracking_linearized(
    tracking: Union[dict, pd.DataFrame],
    ax: plt.axis = None,
    plot:bool=True,
    **kwargs
):
    ax = ax or plt.subplots(figsize=(9, 9))[1]

    x = tracking['global_coord']
    y = np.linspace(1, 0, len(x))

    if not plot:
        ax.scatter(x, y, **kwargs)
    else:
        ax.plot(x,y, **kwargs)


def plot_bouts_1d(
    tracking: Union[dict, pd.DataFrame],
    bouts: pd.DataFrame,
    ax:plt.axis,
    direction:bool = None,
    zorder:int=100,
    lw:float=2,
    alpha:float=1,
    **kwargs,
):
    # select bouts by direction
    if direction is not None:
        bouts = bouts.loc[bouts.direction == direction]

    # get coords
    x = tracking['global_coord']
    y = np.linspace(1, 0, len(x))

    # plot
    for i, bout in bouts.iterrows():
        _x = x[bout.start_frame:bout.end_frame]
        _y = y[bout.start_frame:bout.end_frame]

        ax.plot(_x, _y, color=colors.bout_direction_colors[bout.direction], zorder=zorder, lw=lw, alpha=alpha, **kwargs)
        ax.scatter(_x[0], _y[0], color='white', lw=1, ec=colors.bout_direction_colors[bout.direction], s=30, zorder=101, alpha=.85, **kwargs)
        ax.scatter(_x[-1], _y[-1], color=[.2, .2, .2], lw=1, ec=colors.bout_direction_colors[bout.direction], s=30, zorder=101, alpha=.85, **kwargs)

# ------------------------------------ 2D ------------------------------------ #
def plot_tracking_xy(
    tracking: Union[dict, pd.DataFrame],
    key: str = None,
    skip_frames: int = 1,
    ax: plt.axis = None,
    plot:bool=False,
    **kwargs,
):
    ax = ax or plt.subplots(figsize=(9, 9))[1]

    if key is None:
        if not plot:
            ax.scatter(
                tracking["x"][::skip_frames],
                tracking["y"][::skip_frames],
                color=[0.3, 0.3, 0.3],
                **kwargs,
            )
        else:
            ax.plot(
                tracking["x"][::skip_frames],
                tracking["y"][::skip_frames],
                **kwargs,
            )
    else:
        ax.scatter(
            tracking["x"][::skip_frames],
            tracking["y"][::skip_frames],
            c=tracking[key][::skip_frames],
            **kwargs,
        )

        if "orientation" in key or "angle" in key:
            # draw arrows to mark the angles/colors mapping
            angles = np.linspace(0, 2 * np.pi, 16)
            x = 2 * np.cos(angles[::-1] + np.pi / 2) + 25
            y = 2 * np.sin(angles + np.pi / 2) + 2
            ax.scatter(
                x, y, s=80, zorder=50, c=np.degrees(angles), alpha=1, **kwargs
            )


def plot_bouts_2d(
    tracking: Union[dict, pd.DataFrame],
    bouts: pd.DataFrame,
    ax:plt.axis,
    direction:bool = None,
    zorder:int=100,
    lw:float=2,
    c:str=None,
    **kwargs,
):
    # select bouts by direction
    if direction is not None:
        bouts = bouts.loc[bouts.direction == direction]

    # plot
    for i, bout in bouts.iterrows():
        x = tracking['x'][bout.start_frame:bout.end_frame]
        y = tracking['y'][bout.start_frame:bout.end_frame]

        if c is None:
            color = colors.bout_direction_colors[bout.direction]
        else:
            color = c
        ax.plot(x, y, color=color, zorder=zorder, lw=lw, **kwargs)
        ax.scatter(x[0], y[0], color='white', lw=1, ec=color, s=25, zorder=101, **kwargs)
        ax.scatter(x[-1], y[-1], color=[.2, .2, .2], lw=1, ec=color, s=25, zorder=101, **kwargs)