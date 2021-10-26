# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from data.arena import ROIs_dict
from geometry.vector_utils import smooth_path_vectors

from data import paths

# from draw import colors

import draw
from geometry import Path

# %%
ROI = "T2"
bouts = (
    pd.read_hdf(
        paths.analysis_folder
        / "behavior"
        / "roi_crossings"
        / f"{ROI}_crossings.h5"
    )
    .sort_values("duration")
    .iloc[0:200]
)


# %%
f = plt.figure(figsize=(20, 10))
axes = f.subplot_mosaic(
    """
        AAABBDDFFF
        AAACCEEFFF
    """
)


draw.ROI(
    ROI,
    ax=axes["A"],
    set_ax=True,
    # img_path=r'C:\Users\Federico\Documents\GitHub\pysical_locomotion\draw\hairpin.png'
    img_path="/Users/federicoclaudi/Documents/Github/LocomotionControl/draw/hairpin.png",
)
draw.ROI(
    ROI,
    ax=axes["F"],
    set_ax=True,
    # img_path=r'C:\Users\Federico\Documents\GitHub\pysical_locomotion\draw\hairpin.png'
    img_path="/Users/federicoclaudi/Documents/Github/LocomotionControl/draw/hairpin.png",
)

gcoord_bouts = ROIs_dict[ROI].g_0, ROIs_dict[ROI].g_1
wnd = 3
for i, cross in bouts.iterrows():
    path = Path(cross.x, cross.y)

    velocity, acceleration, tangent = smooth_path_vectors(path, window=wnd)

    time = np.linspace(wnd / 60, cross.duration, len(path) - wnd)

    _acceleration = acceleration.dot(tangent)

    axes["B"].plot(time, velocity.magnitude[wnd:], color="k", alpha=0.2)
    axes["C"].plot(time, _acceleration[wnd:], color="r", alpha=0.2)

    axes["D"].plot(cross.thetadot, color="m", alpha=0.2)
    axes["E"].plot(cross.thetadotdot, color="m", alpha=0.2)

    draw.Tracking.scatter(
        path.x[wnd:],
        path.y[wnd:],
        c=_acceleration[wnd:],
        cmap="bwr",
        vmin=-8,
        vmax=8,
        ax=axes["A"],
    )
    draw.Tracking.scatter(
        path.x[wnd], path.y[wnd], color="k", zorder=200, ax=axes["A"]
    )

    draw.Tracking.scatter(
        path.x[wnd:],
        path.y[wnd:],
        c=cross.thetadotdot[wnd:],
        cmap="bwr",
        vmin=-50,
        vmax=50,
        ax=axes["F"],
    )
    draw.Tracking.scatter(
        path.x[wnd], path.y[wnd], color="k", zorder=200, ax=axes["F"]
    )
_ = axes["C"].axhline(0)

# TODO expand ROIs until acc > 0
# TODO get when acc < 0
# TODO compute averages of e.g. speed, accell...

# TODO make pretty plots including heatmap of neg acc start


# %%

# %%

# %%
