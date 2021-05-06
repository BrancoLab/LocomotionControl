from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from vedo.colors import colorMap

from fcutils.plot.elements import plot_line_outlined

from myterial import (
    indigo,
    salmon,
    blue_grey_darker,
    teal,
    teal_darker,
)


from kinematics.fixtures import BODY_NAMES, PAWS_COLORS


def line(bp1, bp2, ax, frames=None, **kwargs):
    """ deaw a line between body parts
        given a tracking dataframe and selected frames
    """
    if frames is None:
        frames = np.zeros(1)

    try:
        x1 = bp1.x[frames]
        y1 = bp1.y[frames]
        x2 = bp2.x[frames]
        y2 = bp2.y[frames]
    except IndexError:
        x1, y1 = bp1.x, bp1.y
        x2, y2 = bp2.x, bp2.y

    ax.plot([x1, x2], [y1, y2], solid_capstyle="round", **kwargs)


def point(bp, ax, frames=None, **kwargs):
    """
        Draw a scatter point over  a body part
        given a tracking dataframe and selected frames
    """
    if frames is None:
        frames = np.zeros(1)

    try:
        x = bp.x[frames]
        y = bp.y[frames]
    except IndexError:
        # only one frame in data
        x, y = bp.x, bp.y

    ax.scatter(x, y, **kwargs)


def draw_mouse(trial, ax, frames, bps=None, **kwargs):
    """
        Draw a mouse's outline as a polygon
        whose vertices lay at the coordinates
        of a set of bodyparts.

        Given a tracking dataframe and a set of selected frames
        
        Arguments:
            trial: Trial
            ax: plt ax
            frames; list or np array of frames at which to show the mouse
            bps: list of str of body parts names
    """
    # plot body outline
    bps = bps or BODY_NAMES
    patches = []
    for n in frames:
        x = [trial[bp].x[n] for bp in bps]
        y = [trial[bp].y[n] for bp in bps]
        patches.append(Polygon(np.vstack([x, y]).T, True, lw=None, zorder=-5))

    p = PatchCollection(
        patches, alpha=0.1, color=blue_grey_darker, lw=None, zorder=-5
    )
    ax.add_collection(p)

    # plot trajectory
    plot_line_outlined(
        ax,
        trial.body.x[frames[0] : frames[-1]],
        y=trial.body.y[frames[0] : frames[-1]],
        lw=2,
        alpha=1,
        outline_color=[0.8, 0.8, 0.8],
        outline=1,
        color=blue_grey_darker,
        zorder=90,
    )

    # mark start and stop
    n = [frames[0], frames[-1]]
    point(
        trial.body,
        ax,
        n,
        c=[teal_darker, teal],
        lw=1,
        edgecolors=blue_grey_darker,
        alpha=1,
        zorder=100,
    )


def draw_paws_steps_XY(trial, ax, step_starts):
    """
        Plot the position of paws and lines between them
        on the XY plane
    """
    # Plot paws
    for paw, color in PAWS_COLORS.items():
        point(
            trial[paw], ax, step_starts, zorder=1, color=color, s=10,
        )

    # plot paw lines
    line(
        trial.left_hl,
        trial.right_fl,
        ax,
        step_starts,
        color=salmon,
        lw=2,
        zorder=20,
    )
    line(
        trial.right_hl,
        trial.left_fl,
        ax,
        step_starts,
        color=indigo,
        lw=2,
        zorder=20,
    )


def mark_steps(ax, starts, ends, y, scale, noise=0, **kwargs):
    """
        Draw lines to mark when steps start/end

        Y is the height in the plot where the horix lines are drawn
    """
    starts, ends = list(starts), list(ends)

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
