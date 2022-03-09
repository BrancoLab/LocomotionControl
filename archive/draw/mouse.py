import matplotlib.pyplot as plt
import sys
import numpy as np

from myterial import (
    grey_darker,
    grey_dark,
    red_dark,
    purple,
    purple_darker,
    indigo,
    indigo_darker,
)

sys.path.append("./")
from draw import Polygon, Arrow, Dot
from geometry.vector import Vector


COLORS = {
    "left_fl": purple,
    "right_fl": indigo,
    "right_hl": purple_darker,
    "left_hl": indigo_darker,
}


class Mouse:
    def __init__(
        self,
        tracking: dict,
        at_frame: int = 0,
        show_paws: bool = True,
        ax: plt.Axes = None,
    ):
        """
            Given a dictoinary of tracking of the form 
                bp: dict(x=np.ndarray, y=np.ndarray)... 
            this function draws the mouse
        """
        ax = ax or plt.gca()

        self.plot_outline(tracking, at_frame, ax)
        self.plot_body_line(tracking, at_frame, ax)
        self.plot_head(tracking, at_frame, ax)

        if show_paws:
            self.plot_paws(tracking, at_frame, ax)

    def plot_outline(self, tracking: dict, at_frame: int, ax: plt.Axes):
        """
            Once new tracking is availablel, this will draw the mouse outline
            base on shoulders and hips, currenty with paws
        """
        xy = "xy"
        bps = ("left_fl", "right_fl", "right_hl", "left_hl")
        points = [[tracking[bp][c][at_frame] for c in "xy"] for bp in bps]

        Polygon(*points, ax=ax, color=grey_dark, lw=0)

    def plot_body_line(self, tracking: dict, at_frame: int, ax: plt.Axes):
        bps = [("neck", "body"), ("body", "tail_base")]
        for bp1, bp2 in bps:
            ax.plot(
                [tracking[bp1]["x"][at_frame], tracking[bp2]["x"][at_frame]],
                [tracking[bp1]["y"][at_frame], tracking[bp2]["y"][at_frame]],
                lw=4,
                color=grey_darker,
                zorder=2,
            )
        Dot(
            tracking["body"]["x"][at_frame],
            tracking["body"]["y"][at_frame],
            s=150,
            color="k",
        )

    def plot_head(self, tracking: dict, at_frame: int, ax: plt.Axes):
        # get head vectpr
        head_vec = Vector(
            (tracking["snout"]["x"] - tracking["neck"]["x"])[at_frame],
            (tracking["snout"]["y"] - tracking["neck"]["y"])[at_frame],
        )

        # draw an arrow
        Arrow(
            tracking["neck"]["x"][at_frame],
            tracking["neck"]["y"][at_frame],
            head_vec.angle,
            L=2,
            color=red_dark,
            zorder=3,
        )

    def plot_paws(self, tracking, at_frame, ax):
        paws = ("left_fl", "right_fl", "right_hl", "left_hl")
        for paw in paws:
            Dot(
                tracking[paw]["x"][at_frame],
                tracking[paw]["y"][at_frame],
                s=50,
                color=COLORS[paw],
            )

        pairs = (("left_fl", "right_hl"), ("right_fl", "left_hl"))
        for bp1, bp2 in pairs:
            ax.plot(
                [tracking[bp1]["x"][at_frame], tracking[bp2]["x"][at_frame]],
                [tracking[bp1]["y"][at_frame], tracking[bp2]["y"][at_frame]],
                lw=4,
                color=grey_darker,
                zorder=2,
            )


class Mice:
    def __init__(
        self,
        tracking: dict,
        step: int = 10,
        show_paws: bool = True,
        ax: plt.Axes = None,
    ):
        for frame in np.arange(0, len(tracking["body"]["x"]), step):
            Mouse(tracking, at_frame=frame, show_paws=show_paws, ax=ax)


if __name__ == "__main__":
    import pandas as pd

    tracking = pd.read_hdf("./draw/example_tracking.h5", key="hdf").to_dict()

    Mice(tracking)

    plt.show()
