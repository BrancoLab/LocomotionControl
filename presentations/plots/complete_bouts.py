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

from data import paths
from data.data_structures import LocomotionBout, merge_locomotion_bouts

import draw

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

folder = Path(
    r"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
)
recorder.start(
    base_folder=folder.parent, folder_name=folder.name, timestamp=False
)


# %%
# load and clean complete bouts
_bouts = pd.read_hdf(
    paths.analysis_folder / "behavior" / "saved_data" / f"complete_bouts.h5"
)
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
# Plot the whole tracking colored just by gcoord

# plot ROI
f, ax = plt.subplots(figsize=(8, 12))
f._save_name = "gcoord_tracking"
draw.Hairpin()

G = []
for bout in bouts:
    start = np.where(bout.speed > 10)[0][0]
    G.append(bout.gcoord[start:])
G = np.hstack(G)

Y = Y[X < 40]
G = G[X < 40]
X = X[X < 40]


draw.Tracking.scatter(X[::5], Y[::5], c=G[::5], cmap="bone")


recorder.add_figures()

# %%
# Get distance travelled during bout
dists = []
for bout in bouts:
    dists.append(np.sum(bout.speed) / 60)

f, ax = plt.subplots()

draw.Hist(dists, bins=20)


ax.set(xlim=[200, 300])
# %%
# draw hist of bouts durations

f, ax = plt.subplots()

draw.Hist([b.duration for b in bouts], bins=20)


# %%
