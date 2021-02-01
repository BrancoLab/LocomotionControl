from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from vedo.colors import colorMap

from myterial import (
    indigo,
    salmon,
    blue_grey_darker,
    teal,
    teal_darker,
)


from fcutils.plot.elements import plot_line_outlined

# ---------------------------------------------------------------------------- #
#                                  plot utils                                  #
# ---------------------------------------------------------------------------- #


def line(bp1, bp2, ax, tracking, frames, **kwargs):
    """ deaw a line between body parts
        given a tracking dataframe and selected frames
    """
    x1 = tracking[f"{bp1}_x"].values[frames]
    y1 = tracking[f"{bp1}_y"].values[frames]
    x2 = tracking[f"{bp2}_x"].values[frames]
    y2 = tracking[f"{bp2}_y"].values[frames]

    ax.plot([x1, x2], [y1, y2], solid_capstyle="round", **kwargs)


def point(bp, ax, tracking, frames, **kwargs):
    """
        Draw a scatter point over  a body part
        given a tracking dataframe and selected frames
    """
    x = tracking[f"{bp}_x"].values[frames]
    y = tracking[f"{bp}_y"].values[frames]

    ax.scatter(x, y, **kwargs)


def draw_mouse(
    ax, tracking, whole_session_tracking, frames, bps=None, **kwargs
):
    """
        Draw a mouse's outline as a polygon
        whose vertices lay at the coordinates
        of a set of bodyparts.

        Given a tracking dataframe and a set of selected frames
    """
    # plot whole session tracking in the background
    x = whole_session_tracking.body_x.values
    y = whole_session_tracking.body_y.values
    ax.plot(x, y, color=blue_grey_darker, lw=0.76, alpha=0.15, zorder=0)

    bps = bps or (
        "tail_base",
        "left_hl",
        "left_ear",
        "snout",
        "right_ear",
        "right_hl",
    )
    patches = []
    for n in frames:  # range(len(tracking["body_x"])):
        x = [tracking[f"{bp}_x"].values[n] for bp in bps]
        y = [tracking[f"{bp}_y"].values[n] for bp in bps]
        patches.append(Polygon(np.vstack([x, y]).T, True, lw=None, zorder=-5))

    p = PatchCollection(
        patches, alpha=0.3, color=blue_grey_darker, lw=None, zorder=-5
    )
    ax.add_collection(p)

    plot_line_outlined(
        ax,
        tracking["body_x"][frames[0] : frames[-1]],
        y=tracking["body_y"][frames[0] : frames[-1]],
        lw=2,
        alpha=1,
        outline_color=[0.8, 0.8, 0.8],
        outline=1,
        color=blue_grey_darker,
        zorder=90,
    )

    # mark start and stop
    n = [frames[0], frames[-1]]
    ax.scatter(
        tracking["body_x"].iloc[n],
        tracking["body_y"].iloc[n],
        lw=1,
        edgecolors=blue_grey_darker,
        alpha=1,
        c=[teal_darker, teal],
        zorder=100,
    )


def draw_paws_steps(paw_colors, ax, tracking, step_starts, start):

    # Plot paws
    for paw, color in paw_colors.items():
        point(
            paw, ax, tracking, step_starts, zorder=1, color=color, s=10,
        )

    # plot paw lines
    line(
        "left_hl",
        "right_fl",
        ax,
        tracking,
        step_starts,
        color=salmon,
        lw=2,
        zorder=20,
    )
    line(
        "right_hl",
        "left_fl",
        ax,
        tracking,
        step_starts,
        color=indigo,
        lw=2,
        zorder=20,
    )


def mark_steps(ax, starts, ends, y, side, scale, noise=0, **kwargs):
    """
        Draw lines to mark when steps start/end

        Y is the height in the plot where the horix lines are drawn
    """
    starts, ends = list(starts), list(ends)

    # mark which side it is
    ax.text(0, y, side, horizontalalignment="left")

    _color = kwargs.pop("color", None)
    # mark each step
    for n, (start, end) in enumerate(zip(starts, ends)):
        if noise:
            nu = np.random.normal(0, noise)
        else:
            nu = 0

        color = _color or colorMap(n, name="inferno", vmin=0, vmax=len(starts))

        ax.plot([start, end], [y + nu, y + nu], color=color, **kwargs)
        ax.plot(
            [start, start],
            [y - scale + nu, y + scale + nu],
            color=color,
            **kwargs,
        )
        ax.plot(
            [end, end], [y - scale + nu, y + scale + nu], color=color, **kwargs
        )
