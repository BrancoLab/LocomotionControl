# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tpd import recorder
from myterial import pink, blue_dark


from data import paths
from data.data_structures import LocomotionBout
import draw
from geometry import Vector
from geometry.vector_utils import vectors_mean

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=False)

folder = Path(
    r"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
)
recorder.start(
    base_folder=folder.parent, folder_name=folder.name, timestamp=False
)

"""
    Gets the tracking data of a ROI crossing and draws vector fields for the 
    velocity and acceleration vectors
"""

# %%
# ---------------------------------------------------------------------------- #
#                                   data prep                                  #
# ---------------------------------------------------------------------------- #
# load and clean roi crossings
ROI = "T2"
MIN_DUR = 1.5
USE = 100

_bouts = pd.read_hdf(
    paths.analysis_folder / "behavior" / "saved_data" / f"{ROI}_crossings.h5"
).sort_values("duration")

_l = len(_bouts)
_bouts = _bouts.loc[_bouts.duration <= MIN_DUR]
_bouts = _bouts.iloc[:USE]
print(f"Kept {len(_bouts)}/{_l} bouts")

crosses = []
for i, bout in _bouts.iterrows():
    crosses.append(LocomotionBout(bout))


# %%
# ---------------------------------------------------------------------------- #
#                          first plot - sanity checks                          #
# ---------------------------------------------------------------------------- #
"""
    Plot tracking over threshold crossings for sanity checks
"""
colors = dict(velocity=blue_dark, acceleration=pink)

f, axes = plt.subplots(figsize=(16, 10), ncols=2)

for ax in axes:
    draw.ROI(ROI, set_ax=True, ax=ax)
f._save_name = f"vector_field_{ROI}"

all_xy_vectors = {
    "velocity": {},
    "acceleration": {},
}  # stores vectors at each XY position, for averaging
scale_factor = {"velocity": 20, "acceleration": 1}
for vn, (vec_name, xy_vectors) in enumerate(all_xy_vectors.items()):
    ax = axes[vn]

    cross_bins = []
    for cross in crosses:
        # round XY tracking to closest even number
        X, Y = ((cross.x) / 2).round(0) * 2, ((cross.y) / 2).round(0) * 2
        for t, (x, y) in enumerate(zip(X, Y)):
            # add entry to dictionary
            if (x, y) not in xy_vectors.keys():
                xy_vectors[(x, y)] = []

            xy_vectors[(x, y)].append(cross.path[vec_name][t])
            cross_bins.append((x, y))

    # draw stuff
    arrows = dict(x=[], y=[], L=[], theta=[])
    for (x, y), vecs in xy_vectors.items():
        vectors = Vector.from_list(vecs)
        mean_vector = vectors_mean(*vecs)

        ax.scatter(x, y, s=10, color="k", zorder=500)
        arrows["x"].append(x)
        arrows["y"].append(y)
        arrows["theta"].append(mean_vector.angle)
        arrows["L"].append(mean_vector.magnitude)

    draw.Arrows(
        arrows["x"],
        arrows["y"],
        arrows["theta"],
        L=1.5
        * np.clip(arrows["L"], 0, scale_factor[vec_name])
        / scale_factor[vec_name],
        # L=1.5,
        width=1.5,
        ax=ax,
        color=colors[vec_name],
        zorder=1000,
        alpha=1,
    )
    ax.set(title=vec_name)


# recorder.add_figures()


# %%
