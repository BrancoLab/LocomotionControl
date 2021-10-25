import sys

sys.path.append("./")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data import paths
from data.data_utils import convolve_with_gaussian
from draw import colors
from analysis.fixtures import PAWS

import draw
from geometry import Path

#  TODO: add mouse outline?


def plot_roi_crossing(
    crossing: pd.Series, tracking: dict, step: int = 5, highlight: str = None, arrow_scale=.5,
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

    # path = Path(
    #     convolve_with_gaussian(crossing.x, kernel_width=11), 
    #     convolve_with_gaussian(crossing.y, kernel_width=11))
    path = Path(crossing.x, crossing.y)


    f = plt.figure(figsize=(20, 10))
    axes = f.subplot_mosaic(
        """
            AACC
            AADD
            BBEE
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
        L=arrow_scale * path.velocity.magnitude / 30,
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
        L=arrow_scale * 6 * path.acceleration.magnitude / 30,
        step=step,
        color=colors.acceleration,
        ax=axes["A"],
        outline=True,
        alpha=a_alpha,
    )

    draw.Tracking.scatter(
        path.x[::step],
        path.y[::step],
        color='k',
        ax=axes["A"],
        zorder=200,
    )

    # draw speed traces
    time = np.arange(len(path.speed)) / 60
    axes["B"].fill_between(time, 0, path.speed, alpha=0.5, color=colors.speed)
    axes["B"].plot(time, path.speed, lw=4, color=colors.speed,)

    for bp in PAWS:
        axes['C'].plot(tracking[bp]['bp_speed'], color=colors.bodyparts[bp], label=bp)
    axes['C'].plot(tracking['body']['bp_speed'], color=colors.bodyparts['body'], label='body')
    axes['C'].plot(np.mean(np.vstack([tracking[bp]['bp_speed'] for bp in PAWS if 'h' in bp]), 0), color='k', label='mean')

    # clean plots
    axes["A"].legend()
    axes["B"].set(xlabel="time (s)", ylabel="speed (cm/s)")
    axes['C'].legend()
    f.tight_layout()

    return f


if __name__ == "__main__":
    from data.dbase.db_tables import ROICrossing

    bouts = pd.read_hdf(
        paths.analysis_folder
        / "behavior"
        / "roi_crossings"
        / f"T3_crossings.h5"
    ).sort_values('duration').sample(10)
    
    # bout = bouts.loc[bouts.crossing_id == 2638].iloc[0]

    for n, (i, bout) in enumerate(bouts.iterrows()):
        if n > 3: break

        tracking = ROICrossing.get_crossing_tracking(bout.crossing_id)

        # plot_roi_crossing(bout, tracking, step=2)
        plot_roi_crossing(bout, tracking, step=2, highlight='acceleration')

        break

    plt.show()
