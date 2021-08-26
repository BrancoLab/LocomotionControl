import matplotlib.pyplot as plt
from typing import Union
import pandas as pd
import numpy as np


def plot_tracking_xy(
    tracking: Union[dict, pd.DataFrame],
    key: str = None,
    skip_frames: int = 1,
    ax: plt.axis = None,
    **kwargs,
):
    ax = ax or plt.subplots(figsize=(9, 9))[1]

    if key is None:
        ax.scatter(
            tracking["x"][::skip_frames],
            tracking["y"][::skip_frames],
            color=[0.3, 0.3, 0.3],
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
            angles = np.linspace(0, 2 * np.pi, 60)
            x = 2 * np.cos(angles[::-1] + np.pi / 2) + 25
            y = 2 * np.sin(angles + np.pi / 2) + 2
            ax.scatter(
                x, y, s=80, zorder=50, c=np.degrees(angles), alpha=1, **kwargs
            )


def plot_balls_errors(
    x: np.ndarray, y: np.ndarray, yerr: np.ndarray, colors: list, ax: plt.axis
):
    """
        Given a serires of XY values and Y errors it plots a scatter for each XY point and a line
        to mark each Y error
    """
    ax.scatter(x, y, s=150, c=colors, zorder=100, lw=1, ec=[0.3, 0.3, 0.3])

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
