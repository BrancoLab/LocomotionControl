from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

from myterial import blue_grey_darker

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


def draw_mouse(ax, tracking, frames, **kwargs):
    """
        Draw a mouse's outline as a polygon
        whose vertices lay at the coordinates
        of a set of bodyparts.

        Given a tracking dataframe and a set of selected frames
    """
    patches = []
    for n in frames:  # range(len(tracking["body_x"])):
        bps = (
            "tail_base",
            "LH",
            "l_ear",
            "snout",
            "r_ear",
            "RH",
        )
        x = [tracking[f"{bp}_x"].values[n] for bp in bps]
        y = [tracking[f"{bp}_y"].values[n] for bp in bps]
        patches.append(Polygon(np.vstack([x, y]).T, True, lw=None))

    p = PatchCollection(patches, alpha=0.3, color=blue_grey_darker, lw=None)
    ax.add_collection(p)

    ax.plot(
        tracking["body_x"][frames[0] :],
        tracking["body_y"][frames[0] :],
        lw=3,
        alpha=0.4,
        color=[0.4, 0.4, 0.4],
        zorder=-5,
    )


def mark_steps(ax, starts, ends, y, side, scale, noise=0, **kwargs):
    """
        Draw lines to mark when steps start/end

        Y is the height in the plot where the horix lines are drawn
    """
    starts, ends = list(starts), list(ends)

    # mark which side it is
    ax.text(0, y, side, horizontalalignment="left")
    # mark each step
    for start, end in zip(starts, ends):
        if noise:
            nu = np.random.normal(0, noise)
        else:
            nu = 0

        ax.plot([start, end], [y + nu, y + nu], **kwargs)
        ax.plot([start, start], [y - scale + nu, y + scale + nu], **kwargs)
        ax.plot([end, end], [y - scale + nu, y + scale + nu], **kwargs)
