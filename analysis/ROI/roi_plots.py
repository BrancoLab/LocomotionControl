import sys

sys.path.append("./")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data import paths, colors

import draw
from geometry import Path

#  TODO: add mouse outline?


def plot_roi_crossing(
    crossing: pd.Series, step: int = 5, highlight: str = None
) -> plt.Figure:
    """
        Plots an entire ROI crossing, tracking, vectors and mouse
    """
    if highlight == "velocity":
        v_alpha, a_alpha = 1, 0.4
    elif highlight == "acceleration":
        v_alpha, a_alpha = 0.4, 1
    else:
        v_alpha, a_alpha = 1, 1

    path = Path(crossing.x, crossing.y)

    f = plt.figure(figsize=(8, 10))
    axes = f.subplot_mosaic(
        """
            AA
            AA
            BB
        """
    )
    f._save_name = (
        f"{crossing.roi}_{crossing.crossing_id}" + ""
        if highlight is None
        else highlight
    )

    # draw main crossing plot
    draw.ROI(crossing.roi, ax=axes["A"], set_ax=True)
    draw.Tracking(path.x, path.y, lw=0.5, color="k", ax=axes["A"])

    # draw velocity and acceleartion vectors
    draw.Arrows(
        path.x,
        path.y,
        path.velocity.angle,
        label="velocity",
        L=path.velocity.magnitude / 30,
        step=step,
        color=colors.velocity,
        ax=axes["A"],
        outline=True,
        alpha=v_alpha,
    )

    draw.Arrows(
        path.x,
        path.y,
        path.acceleration.angle,
        label="acceleration",
        L=2 * path.acceleration.magnitude / 30,
        step=step,
        color=colors.acceleration,
        ax=axes["A"],
        outline=True,
        alpha=a_alpha,
    )

    # draw speed traces
    time = np.arange(len(path.speed)) / 60
    axes["B"].fill_between(time, 0, path.speed, alpha=0.5, color=colors.speed)
    axes["B"].plot(time, path.speed, lw=4, color=colors.speed)

    axes["A"].legend()
    axes["B"].set(xlabel="time (s)", ylabel="speed (cm/s)")
    f.tight_layout()

    return f


if __name__ == "__main__":

    bout = pd.read_hdf(
        paths.analysis_folder
        / "behavior"
        / "roi_crossings"
        / f"T2_crossings.h5"
    ).iloc[0]

    plot_roi_crossing(bout, step=4)

    # plot_roi_crossing(bout, highlight='velocity')
    # plot_roi_crossing(bout, highlight='acceleration')

    plt.show()
