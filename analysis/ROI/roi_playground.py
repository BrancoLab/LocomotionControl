# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from data import paths

from draw import colors

import draw
from geometry import Path, Vector
from geometry.vector_utils import vectors_mean

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

for i, cross in bouts.iterrows():
    path = Path(cross.x, cross.y)

    time = np.linspace(0, cross.duration, len(path))

    axes["B"].plot(time, cross.speed, color="k", alpha=0.2)
    axes["C"].plot(time, cross.acceleration, color="r", alpha=0.2)

    axes["D"].plot(cross.thetadot, color="m", alpha=0.2)
    axes["E"].plot(cross.thetadotdot, color="m", alpha=0.2)

    draw.Tracking.scatter(
        path.x,
        path.y,
        c=cross.acceleration,
        cmap="bwr",
        vmin=-8,
        vmax=8,
        ax=axes["A"],
    )
    draw.Tracking.scatter(
        path.x[0], path.y[0], color="k", zorder=200, ax=axes["A"]
    )

    draw.Tracking.scatter(
        path.x,
        path.y,
        c=cross.thetadotdot,
        cmap="PiYG",
        vmin=-50,
        vmax=50,
        ax=axes["F"],
    )
    draw.Tracking.scatter(
        path.x[0], path.y[0], color="k", zorder=200, ax=axes["F"]
    )
_ = axes["C"].axhline(0)

# TODO expand ROIs until acc > 0
# TODO get when acc < 0
# TODO compute averages of e.g. speed, accell...

# TODO make pretty plots including heatmap of neg acc start


# %%
# attempt at a vector field visualizatoin

f, axes = plt.subplots(figsize=(20, 10), ncols=2)

for ax in axes:
    draw.ROI(
        ROI,
        ax=ax,
        set_ax=True,
        # img_path=r'C:\Users\Federico\Documents\GitHub\pysical_locomotion\draw\hairpin.png'
        img_path="/Users/federicoclaudi/Documents/Github/LocomotionControl/draw/hairpin.png",
    )

all_xy_vectors = {
    "velocity": {},
    "acceleration": {},
}  # stores vectors at each XY position, for averaging
scale_factor = {"velocity": 1 / 80, "acceleration": 1 / 5}
for vn, (vec_name, xy_vectors) in enumerate(all_xy_vectors.items()):
    ax = axes[vn]

    for n, (i, cross) in enumerate(bouts.iterrows()):
        if n > 200:
            break

        path = Path(cross.x, cross.y)

        cross_bins = []
        for t, (x, y) in enumerate(zip(path.x.round(0), path.y.round(0))):
            if (x, y) in cross_bins:
                continue  # avoid repeated sampling of same location

            # add entry to dictionary
            if (x, y) not in xy_vectors.keys():
                xy_vectors[(x, y)] = []

            xy_vectors[(x, y)].append(path[vec_name][t])
            cross_bins.append((x, y))

    # draw stuff
    for (x, y), vecs in xy_vectors.items():
        ax.scatter(x, y, s=10, color="k", zorder=500)
        x = [x + 0.5] * len(vecs)
        y = [y + 0.5] * len(vecs)

        vectors = Vector.from_list(vecs)
        mean_vector = vectors_mean(*vecs)

        # draw.Arrows(x, y, vectors.angle, L=0.5, width=0.75, alpha=0.5, color='k', ax=ax)
        draw.Arrow(
            x[0],
            y[0],
            mean_vector.angle,
            L=mean_vector.magnitude * scale_factor[vec_name],
            width=2,
            ax=ax,
            color=colors.variables[vec_name],
            zorder=1000,
            alpha=1,
        )
    ax.set(title=vec_name)


# %%

# %%
