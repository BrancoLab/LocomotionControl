# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from fcutils.plot.figure import clean_axes

from myterial import indigo_dark, pink, blue_grey
from myterial.utils import make_palette

from analysis.load import load_complete_bouts
from data.data_utils import merge_locomotion_bouts
from kinematics.track import extract_track_from_image
from kinematics import track_cordinates_system as TCS
import draw

"""
    Plot the so callled g-g plot: max longitudinal and lateral acceleration 
    at different speeds (data pooled across complete locomotion bouts).


    Then it also plots v_max = \sqrt{\frac{acc_lat}{k}} along the track
"""

# %%
# load and clean complete bouts
bouts = load_complete_bouts(keep=100, window=1)

# merge bouts and get speed, longitudinal and lateral accelerations
_, _, S, A, _, _, _, LonA, LatA = merge_locomotion_bouts(bouts)


# Put the data in a dataframe so that it can be binned by speed
data = pd.DataFrame(dict(speed=S, longitudinal=LonA, lateral=LatA))

# bin the data by speed
MIN_SPEED = 30
MAX_SPEED = 80
BINS = np.linspace(MIN_SPEED, MAX_SPEED, 6)
data["binned_speed"] = pd.cut(
    data.speed, bins=BINS, labels=BINS.astype(np.int32)[:-1]
)

# filter empty bins
data = data.loc[~np.isnan(list(data.binned_speed))]
speeds = sorted(list(data["binned_speed"].unique()))
delta = int(np.diff(BINS)[0])
speeds_labels = [f"{b}-{b+delta} cm/s" for b in BINS.astype(np.int32)[:-1]]

# get themin/max accelerations for each bin
binned = {
    "speed": [],
    "longitudinal_low": [],
    "longitudinal_high": [],
    "lateral_low": [],
    "lateral_high": [],
}
for speed in speeds:
    speed_data = data.loc[data.binned_speed == speed]
    binned["speed"].append(speed)
    binned["longitudinal_low"].append(
        np.quantile(speed_data.longitudinal, 0.025)
    )
    binned["longitudinal_high"].append(
        np.quantile(speed_data.longitudinal, 0.975)
    )
    binned["lateral_low"].append(np.quantile(speed_data.lateral, 0.025))
    binned["lateral_high"].append(np.quantile(speed_data.lateral, 0.975))


binned = pd.DataFrame(binned)

to_print = data.groupby("binned_speed").count()
to_print["speed labels"] = speeds_labels
print(to_print[["speed", "speed labels"]])
# %%
# ----------------------------------- plot ----------------------------------- #


# prepare figure
f = plt.figure(figsize=(13.5, 9))
axes = f.subplot_mosaic(
    """
        AABBCCGGG
        DDEEFFGGG
    """
)

speeds_axes = "ABCDE"

for ax in speeds_axes + "G":
    axes[ax].axvline(0, lw=2, color="k")
    axes[ax].axhline(0, lw=2, color="k")

colors = make_palette(indigo_dark, pink, 4)


#  plot mean values
for n, speed in enumerate(speeds):
    ax = axes[speeds_axes[n]]
    speed_data = data.loc[data.binned_speed == speed]

    # plot max longitudinal and lateral acceleration
    ax.scatter(
        [0, 0, binned.iloc[n].lateral_low, binned.iloc[n].lateral_high],
        [
            binned.iloc[n].longitudinal_low,
            binned.iloc[n].longitudinal_high,
            0,
            0,
        ],
        s=100,
        lw=1,
        ec="k",
        zorder=100,
        c=colors,
    )
    ax.axhline(
        binned.iloc[n].longitudinal_low,
        lw=2,
        ls=":",
        zorder=90,
        color=colors[0],
    )
    ax.axhline(
        binned.iloc[n].longitudinal_high,
        lw=2,
        ls=":",
        zorder=90,
        color=colors[1],
    )
    ax.axvline(
        binned.iloc[n].lateral_low, lw=2, ls=":", zorder=90, color=colors[2]
    )
    ax.axvline(
        binned.iloc[n].lateral_high, lw=2, ls=":", zorder=90, color=colors[3]
    )

    # draw heatmap
    sns.kdeplot(
        data=speed_data,
        x="lateral",
        y="longitudinal",
        ax=ax,
        levels=2,
        fill=True,
        alpha=0.75,
        color=blue_grey,
        bw=0.4,
    )

    # style ax
    if n == 0:
        _ = ax.set(
            label="lateral acceleration", ylabel="longitudinal acceleration"
        )
    _ = ax.set(title=f"Speed: {speeds_labels[n]}", xlim=[-4, 4], ylim=[-6, 6])

# plot all GG contours in the same plot
_ = sns.kdeplot(
    data=data,
    x="lateral",
    y="longitudinal",
    hue=data.binned_speed,
    palette="magma",
    levels=1,
    linewidths=2,
    ax=axes["G"],
    bw=0.5,
    zorder=200,
)

# plot normalized accelerations
axes["F"].plot(
    binned.speed,
    np.abs(binned.longitudinal_low) / np.abs(binned.longitudinal_low).max(),
    "-o",
    lw=2,
    markerfacecolor="white",
    ms=10,
    color=colors[0],
    label="long. min",
)
axes["F"].plot(
    binned.speed,
    np.abs(binned.longitudinal_high) / np.abs(binned.longitudinal_high).max(),
    "-o",
    lw=2,
    markerfacecolor="white",
    ms=10,
    color=colors[1],
    label="long. max",
)
axes["F"].plot(
    binned.speed,
    np.abs(binned.lateral_low) / np.abs(binned.lateral_low).max(),
    "-o",
    lw=2,
    markerfacecolor="white",
    ms=10,
    color=colors[2],
    label="lat. min",
)
axes["F"].plot(
    binned.speed,
    np.abs(binned.lateral_high) / np.abs(binned.lateral_high).max(),
    "-o",
    lw=2,
    markerfacecolor="white",
    ms=10,
    color=colors[3],
    label="lat. max",
)
axes["F"].legend()
_ = axes["F"].set(
    ylabel="norm. acceleration",
    xlabel="speed (cm/s)",
    xticks=binned.speed,
    yticks=[0, 0.5, 1],
)

clean_axes(f)
f.tight_layout()


# %%
(
    left_line,
    center_line,
    right_line,
    left_to_track,
    center_to_track,
    right_to_track,
    control_points,
) = extract_track_from_image(
    points_spacing=1, restrict_extremities=False, apply_extra_spacing=False,
)


# project bouts to center line coordinates
centered_bouts = [
    TCS.path_to_track_coordinates_system(center_line, bout.body)
    for bout in bouts
]


# %%
"""
    Plot v_max = \sqrt{\frac{acc_lat}{k}} along the track
"""

f = plt.figure(figsize=(15, 12))
axes = f.subplot_mosaic(
    """
        AAABB
        AAACC
    """
)


# draw the curvature
_ = axes["B"].plot(
    center_line.comulative_distance, center_line.curvature, lw=4, zorder=100
)
axes["B"].set(xticks=[], ylabel="curvature")

# draw single bouts speeds
for bout, centered_bout in zip(bouts, centered_bouts):
    # tracking
    draw.Tracking(bout.body.x, bout.body.y, lw=0.5, ax=axes["A"])

    # curvature
    axes["B"].plot(
        bout.body.comulative_distance,
        bout.body.curvature,
        color="k",
        alpha=0.25,
    )

    # speed
    axes["C"].plot(centered_bout.x, bout.body.speed, color="k", alpha=0.25)

# draw the v_max
for y_max in [2.5]:
    vmax = np.sqrt((y_max * 60) / (center_line.curvature))
    vmax[np.isinf(vmax)] = np.nan
    # vmax = vmax * VMAX_FACTOR + VMAX_SHIFT

    axes["C"].plot(
        center_line.comulative_distance, vmax, label=f"y_max: {y_max}", lw=3
    )

# draw the track
draw.Hairpin(ax=axes["A"], set_ax=True)
draw.Tracking(left_line.x, left_line.y, ax=axes["A"])
draw.Tracking(center_line.x, center_line.y, ls="--", ax=axes["A"])
draw.Tracking.scatter(
    center_line.x,
    center_line.y,
    c=center_line.curvature,
    ax=axes["A"],
    s=75,
    ec="k",
    lw=0.5,
    zorder=100,
)

draw.Tracking(right_line.x, right_line.y, ax=axes["A"])

axes["B"].set(ylim=[0, 0.8])
axes["C"].legend()
_ = axes["C"].set(
    ylim=[0, 125], xlabel="path distance (cm)", ylabel="speed (cm/s)"
)
f.tight_layout()


# %%
plt.plot(bout.body.velocity.magnitude)
plt.plot(np.cumsum(bout.body.longitudinal_acceleration / 60))
# %%
plt.show()
