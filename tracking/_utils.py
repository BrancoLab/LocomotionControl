from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from vedo.colors import colorMap

from myterial import (
    indigo,
    salmon,
    blue_grey_darker,
    pink,
    pink_darker,
)

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


def draw_mouse(ax, tracking, frames, bps=None, **kwargs):
    """
        Draw a mouse's outline as a polygon
        whose vertices lay at the coordinates
        of a set of bodyparts.

        Given a tracking dataframe and a set of selected frames
    """
    bps = bps or ("tail_base", "LH", "l_ear", "snout", "r_ear", "RH",)
    patches = []
    for n in frames:  # range(len(tracking["body_x"])):
        x = [tracking[f"{bp}_x"].values[n] for bp in bps]
        y = [tracking[f"{bp}_y"].values[n] for bp in bps]
        patches.append(Polygon(np.vstack([x, y]).T, True, lw=None))

    p = PatchCollection(patches, alpha=0.1, color=blue_grey_darker, lw=None)
    ax.add_collection(p)

    ax.plot(
        tracking["body_x"][frames[0] :],
        tracking["body_y"][frames[0] :],
        lw=3,
        alpha=0.4,
        color=[0.4, 0.4, 0.4],
        zorder=-5,
    )


def draw_paws_steps(paw_colors, ax, tracking, step_starts, start):

    # Plot paws
    for paw, color in paw_colors.items():
        point(
            paw, ax, tracking, step_starts, zorder=1, color=color, s=50,
        )

    # plot paw lines
    line(
        "LH", "RF", ax, tracking, step_starts, color=salmon, lw=2, zorder=2,
    )
    line(
        "RH", "LF", ax, tracking, step_starts, color=indigo, lw=2, zorder=2,
    )
    line(
        "tail_base",
        "snout",
        ax,
        tracking,
        step_starts,
        color=pink,
        lw=3,
        zorder=2,
    )
    line(
        "tail_base",
        "snout",
        ax,
        tracking,
        np.arange(start, len(tracking["body_x"])),
        color=pink_darker,
        lw=1,
        zorder=-2,
        alpha=0.2,
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
