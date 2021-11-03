# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from tpd import recorder

from data import paths
from data.data_structures import LocomotionBout, merge_locomotion_bouts

import draw
from kinematics import time
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

# folder = Path(
#     r"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
# )
# recorder.start(
#     base_folder=folder.parent, folder_name=folder.name, timestamp=False
# )


# %%
# load and clean complete bouts
_bouts = pd.read_hdf(
    paths.analysis_folder / "behavior" / "saved_data" / f"complete_bouts.h5"
).sort_values("duration")
_bouts = _bouts.loc[_bouts.duration < 10]
# _bouts = _bouts.iloc[:200]
print(f"Kept {len(_bouts)} bouts")

bouts = []
for i, bout in _bouts.iterrows():
    bouts.append(LocomotionBout(bout))

# merge bouts for heatmaps
X, Y, S, A, T, AV, AA = merge_locomotion_bouts(bouts)

# %%


f = plt.figure(figsize=(20, 20))
f._save_name = "complete_bouts"
axes = f.subplot_mosaic(
    """
        AABC
        AADE
    """
)

for ax in "ABCDE":
    axes[ax].axis("equal")
    draw.Hairpin(ax=axes[ax], set_ax=True if ax == "A" else False)

    if ax != "A":
        axes[ax].axis("equal")
        axes[ax].axis("off")

# draw tracking traces
for bout in bouts:
    draw.Tracking(bout.x + 0.5, bout.y + 0.5, ax=axes["A"])

# draw speed heatmap
draw.Tracking.heatmap(
    X + 0.5,
    Y + 0.5,
    c=S,
    gridsize=(40, 60),
    ax=axes["B"],
    vmin=0,
    vmax=80,
    mincnt=10,
    colorbar=True,
    cmap="inferno",
)

# draw acceleration heatmap
draw.Tracking.heatmap(
    X + 0.5,
    Y + 0.5,
    c=A,
    gridsize=(40, 60),
    ax=axes["D"],
    vmin=-3,
    vmax=3,
    mincnt=10,
    colorbar=True,
    cmap="bwr",
)

# draw angular velocity heatmap
draw.Tracking.heatmap(
    X + 0.5,
    Y + 0.5,
    c=AV,
    gridsize=(40, 60),
    ax=axes["C"],
    vmin=-400,
    vmax=400,
    mincnt=10,
    colorbar=True,
    cmap="PRGn",
)

# draw angular acceleration heatmap
draw.Tracking.heatmap(
    X + 0.5,
    Y + 0.5,
    c=AA,
    gridsize=(40, 60),
    ax=axes["E"],
    vmin=-30,
    vmax=30,
    mincnt=10,
    colorbar=True,
    cmap="PRGn",
)


axes["B"].set(title=r"$speed -- \frac{cm}{s}$")
axes["D"].set(title=r"$acceleration -- \frac{cm}{s^2}$")
axes["C"].set(title=r"$angular velocity -- \frac{deg}{s}$")
_ = axes["E"].set(title=r"$angular acceleration -- \frac{deg}{s^2}$")

recorder.add_figures()


# %%
# Plot 2D tracking on arena
from geometry import Path

bouts = _bouts.loc[_bouts.start_roi==0]

paths = time.time_rescale([Path(bout.x, bout.y) for i,bout in bouts.iterrows()])


f, ax = plt.subplots(figsize=(8, 12))

draw.Hairpin()
for bout in paths:
    draw.Tracking(bout.x + 0.5, bout.y + 0.5)

mean_trajectory = time.average_xy_trajectory(paths, rescale=True)

_ = draw.Tracking(mean_trajectory.x, mean_trajectory.y, lw=3, color='salmon')


# %%
import numpy as np
from geometry import Path
paths = [Path(bout.x, bout.y) for i,bout in bouts.iterrows()]

cut_paths = []
for bout in paths:
    moved = np.where(bout.x < 17)[0][0]
    arrived = np.where(bout.x < 8)[0][0]
    
    cut_paths.append(Path(bout.x[moved:arrived], bout.y[moved:arrived]))

scaled_paths = time.time_rescale(cut_paths)


X = np.mean(np.vstack([path.x for path in scaled_paths]), 0)
Y = np.mean(np.vstack([path.y for path in scaled_paths]), 0)

plt.plot(X, Y)
draw.Hairpin()


