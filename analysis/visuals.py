import matplotlib.pyplot as plt
from typing import Union
import pandas as pd
import numpy as np
from loguru import logger


from fcutils.plot.distributions import plot_kde
from fcutils.plot.elements import plot_mean_and_error
from myterial import blue_grey, blue_grey_dark, grey_darker, pink_dark, blue

from data import colors, data_utils
from analysis._visuals import get_window_ticks


# ----------------------------------- misc ----------------------------------- #

def regplot(data:Union[pd.DataFrame, pd.Series, dict], x:str, y:str, ax:plt.axis, scatter_sample:int=10, **kwargs):
    ax.scatter(data[x][::scatter_sample], data[y][::scatter_sample], **kwargs)


def plot_balls_errors(
    x: np.ndarray, y: np.ndarray, yerr: np.ndarray, ax: plt.axis,s:int=150, colors: Union[list, str]=None
):
    """
        Given a serires of XY values and Y errors it plots a scatter for each XY point and a line
        to mark each Y error
    """
    if colors is None:
        colors = [blue_grey] * len(x)
    elif isinstance(colors, str):
        colors = [colors] * len(x)

    ax.scatter(x, y, s=s, c=colors, zorder=100, lw=1, ec=[0.3, 0.3, 0.3])
    ax.plot(x, y, lw=3, color=colors[0], zorder=-1)

    if yerr is not None:
        for n in range(len(x)):
            ax.plot(
                [x[n], x[n]],
                [y[n] - yerr[n], y[n] + yerr[n]],
                lw=4,
                color=[0.3, 0.3, 0.3],
                zorder=96,
                solid_capstyle="round",
            )
            ax.plot(
                [x[n], x[n]],
                [y[n] - yerr[n], y[n] + yerr[n]],
                lw=2,
                color=colors[n],
                zorder=98,
                solid_capstyle="round",
            )


def plot_bin_x_by_y(
                data:pd.DataFrame,
                x:str,
                y:str,
                ax:plt.axis,
                bins:Union[int,
                np.ndarray]=10,
                as_counts:bool=False,
                with_errors:bool=True, 
                min_count:int=0,
                **kwargs):
    '''
        Bin the values of a column X of a dataframe based on the values of
        another column Y and plot as a balls and errors plot
    '''
    # bin
    bins, means, errors, counts = data_utils.bin_x_by_y(data, x, y, bins=bins, min_count=min_count)

    # plot
    if not as_counts:
        plot_balls_errors(
            bins, means, errors if with_errors else None, ax, **kwargs
        )
    else:
        plot_balls_errors(
            bins, counts, None, ax, **kwargs
        )

def plot_aligned(
    x: np.ndarray,
    indices:Union[np.ndarray, list],
    ax:plt.axis,
    mode:str,
    window:int=120,
    mean_kwargs:dict = None,
    **kwargs,
):
    '''
        Given a 1d array and a series of indices it plots the values of 
        the array aligned to the timestamps.
    '''
    pre, aft = int(window/2), int(window/2)
    mean_kwargs = mean_kwargs or dict(lw=4, zorder=100, color=pink_dark)

    # get plotting params
    if mode == 'pre':
        pre_c, pre_lw = kwargs.pop('color', 'salmon'), kwargs.pop('lw', 2)
        aft_c, aft_lw = blue_grey, 1
        ax.axvspan(0, pre, fc=blue, alpha=.1, zorder=-20)
    else:
        aft_c, aft_lw = kwargs.pop('color', 'salmon'), kwargs.pop('lw', 2)
        pre_c, pre_lw = blue_grey, 1
        ax.axvspan(aft, window, fc=blue, alpha=.1, zorder=-20)


    # plot each trace 
    X = []  # collect data to plot mean
    for idx in indices:
        x_pre = x[idx-pre:idx]
        x_aft = x[idx-1:idx+aft]

        if len(x_pre) != pre or len(x_aft) != aft+1:
            logger.warning(f'Could not plot data aligned to index: {idx}')
            continue
        X.append(x[idx-pre:idx+aft])

        ax.plot(x_pre, color=pre_c, lw=pre_lw, **kwargs)
        ax.plot(np.arange(aft+1) + aft, x_aft, color=aft_c, lw=aft_lw, **kwargs)

    # plot mean and line
    X = np.vstack(X)
    plot_mean_and_error(np.mean(X, axis=0), np.std(X, axis=0), ax, **mean_kwargs)
    ax.axvline(pre, lw=2, color=blue_grey_dark, zorder=-1)

    ax.set(**get_window_ticks(window, shifted=False))

def plot_heatmap_2d(
    data:pd.DataFrame,
    key:str=None,
    ax:plt.axis=None,
    x_key:str='x',
    y_key:str='y',
    cmap:str='inferno',
    vmin:int=0,
    vmax:int=100,
    gridsize:int=30,
    mincnt:int = 1,
    **kwargs,
):
    # bin data in 2d
    ax.hexbin(data[x_key], data[y_key], data[key] if key is not None else None, cmap=cmap, gridsize=gridsize, vmin=vmin, vmax=vmax, mincnt=mincnt, **kwargs)



# ---------------------------------------------------------------------------- #
#                                     EPHSY                                    #
# ---------------------------------------------------------------------------- #
def plot_probe_electrodes(
    rsites: pd.DataFrame, ax: plt.axis, TARGETS: list = [], annotate_every: int=5, x_shift:bool=True, s:int=25, lw:float=0.25,
):
    x = np.ones(len(rsites)) * 1.025
    if x_shift:
        x[::2] = 0.925
        x[2::4] = 0.975
        x[1::4] = 1.075

    if TARGETS is not None:
        colors = [
            rs.color
            if rs.brain_region in TARGETS
            else ([0.3, 0.3, 0.3] if rs.color == "k" else blue_grey)
            for i, rs in rsites.iterrows()
        ]
    else:
        colors = [rs.color for i, rs in rsites.iterrows()]

    ax.scatter(
        x,
        rsites.probe_coordinates,
        s=s,
        lw=lw,
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

def plot_raster(spikes:np.ndarray, events:Union[np.ndarray, list], ax:plt.axis, window:int=120, s=5, color=grey_darker, kde:bool=True, bw:int=6, **kwargs):
    '''
        Plots a raster plot of spikes aligned to timestamps events

        It assumes that all event and spike times are in frames and framerate is 60
    '''
    half_window, quarter_window = window/2, window/4
    yticks_step = int(len(events) / 8)
    X = []
    for n, event in enumerate(events):
        event_spikes = spikes[(spikes >= event - half_window) & (spikes <= event + half_window)] - event
        X.extend(list(event_spikes))
        y = np.ones_like(event_spikes) * n
        ax.scatter(event_spikes, y, s=5, color=color, **kwargs)
    ax.axvline(0, ls=':', color='k', lw=.75)

    # plot KDE
    if kde:
        plot_kde(ax=ax, z=-len(events)/4, data=X, normto=len(events)/5, color=blue_grey_dark, kde_kwargs=dict(bw=bw, cut=0), alpha=.6, invert=False)

    # set x axis properties
    ax.set(yticks=np.arange(0, len(events), yticks_step), ylabel='event number', **get_window_ticks(window))


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


# ---------------------------------------------------------------------------- #
#                                     BOUTS                                    #
# ---------------------------------------------------------------------------- #

def plot_bouts_2d(
    tracking: Union[dict, pd.DataFrame],
    bouts: pd.DataFrame,
    ax:plt.axis,
    direction:bool = None,
    zorder:int=100,
    lw:float=2,
    c:str=None,
    unit:pd.Series=None,
    **kwargs,
):
    # select bouts by direction
    if direction is not None:
        bouts = bouts.loc[bouts.direction == direction]

    # plot
    for i, bout in bouts.iterrows():
        # prepare data
        x = tracking['x'][bout.start_frame:bout.end_frame]
        y = tracking['y'][bout.start_frame:bout.end_frame]

        if c is None:
            color = colors.bout_direction_colors[bout.direction]
        else:
            color = c
        
        # plot tracking
        ax.plot(x, y, color=color, zorder=zorder, lw=lw, **kwargs)

        if unit is None:
            # mark start and end
            ax.scatter(x[0], y[0], color='white', lw=1, ec=color, s=25, zorder=101, **kwargs)
            ax.scatter(x[-1], y[-1], color=[.2, .2, .2], lw=1, ec=color, s=25, zorder=101, **kwargs)
        else:
            # mark unit spikes
            spikes = unit.spikes[(unit.spikes > bout.start_frame)&(unit.spikes < bout.end_frame)]
            ax.scatter(
                tracking['x'][spikes],
                tracking['y'][spikes],
                s=15, zorder=101, color=unit.color
            )


def plot_bouts_aligned(
    tracking: Union[dict, pd.DataFrame],
    bouts: pd.DataFrame,
    ax:plt.axis,
    color:str=blue_grey,
    **kwargs,
):
    '''
        Aligns bouts such that they start from the same position and with the same
        orientation.
    '''
    raise NotImplementedError('this doesnt work')
    keys = ['x', 'y', 'speed', 'orientation', 'angular_velocity']

    for i, bout in bouts.iterrows():
        xy = np.vstack(tracking[['x', 'y']].values).T[bout.start_frame:bout.end_frame]

        # center
        xy -= xy[0]

        # rotate
        # R = coordinates.R(tracking['orientation'][bout.start_frame])
        # xy = (R.T @ xy.T).T
        # xy = xy[:20]

        ax.plot(xy[:, 0], xy[:, 1], color=color, **kwargs)


def plot_bouts_x(
    tracking_data:pd.DataFrame,
    bouts:pd.DataFrame,
    ax:plt.axis,
    variable:str,
    color:str=blue_grey,
    **kwargs
):
    '''
        Plots a variable from the tracking data for each bout
    '''
    for i, bout in bouts.iterrows():
        ax.plot(tracking_data[variable][bout.start_frame:bout.end_frame], color=color, **kwargs)

def plot_bouts_x_by_y(
    tracking_data:pd.DataFrame,
    bouts:pd.DataFrame,
    ax:plt.axis,
    x:str,
    y:str,
    color:str=blue_grey,
    **kwargs
):
    '''
        Plots two tracking variables one against the other for each bout
    '''
    for i, bout in bouts.iterrows():
        ax.plot(tracking_data[x][bout.start_frame:bout.end_frame], tracking_data[y][bout.start_frame:bout.end_frame], color=color, **kwargs)



def plot_bouts_heatmap_2d(
    tracking_data:pd.DataFrame,
    bouts:pd.DataFrame,
    var:str,
    ax:plt.axis,
    **kwargs
):
    # stack the data for each bout
    data = dict(x=[], y=[], var=[])
    for i, bout in bouts.iterrows():
        data['x'].extend(list(tracking_data.x[bout.start_frame:bout.end_frame]))
        data['y'].extend(list(tracking_data.y[bout.start_frame:bout.end_frame]))
        data['var'].extend(list(tracking_data[var][bout.start_frame:bout.end_frame]))

    # plot
    plot_heatmap_2d(pd.DataFrame(data), 'var', ax, **kwargs)

