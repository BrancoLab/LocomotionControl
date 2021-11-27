# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import matplotlib.pyplot as plt

from analysis.load import load_complete_bouts
from data.data_structures import merge_locomotion_bouts
import draw

"""
    Plots complete bouts through the arena, heatmaps
    of speed and velocity and angular accelration and velocity
"""
# load and clean complete bouts
bouts = load_complete_bouts(window=5)

# merge bouts for heatmaps
X, Y, S, A, T, AV, AA = merge_locomotion_bouts(bouts)

# %%
# ? Plot heatmaps
f = plt.figure(figsize=(12, 12))
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
    draw.Tracking(bout.body.x + 0.5, bout.body.y + 0.5, ax=axes["A"])

# draw speed heatmap
draw.Tracking.heatmap(
    X + 0.5,
    Y + 0.5,
    c=S,
    gridsize=(40, 60),
    ax=axes["B"],
    vmin=0,
    vmax=80,
    mincnt=3,
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
    mincnt=3,
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
    mincnt=3,
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
    mincnt=3,
    colorbar=True,
    cmap="PRGn",
)


axes["B"].set(title=r"$speed -- \frac{cm}{s}$")
axes["D"].set(title=r"$acceleration -- \frac{cm}{s^2}$")
axes["C"].set(title=r"$angular velocity -- \frac{deg}{s}$")
_ = axes["E"].set(title=r"$angular acceleration -- \frac{deg}{s^2}$")

plt.show()

# %%
