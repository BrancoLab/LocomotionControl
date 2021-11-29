# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from loguru import logger
import seaborn as sns


from kino.locomotion import Locomotion
from myterial import (
    light_blue,
    indigo_dark,
    blue_darker,
    pink,
)


import draw
from draw import colors
from data.arena import ROIs_dict
from kinematics import track
from kinematics import track_cordinates_system as TCS
from analysis.load import load_complete_bouts
from data.data_structures import AtPoint


"""
    Analysis of the turn kinematics with data from complete bouts but looking only at single turns.


    1. define AtPoint calss to collect kinematics at same time point across locomotion instances
    2. load data (slow) + trim for selected ROI
    3. draw tracking data
    4. get AtPoint for ROI crossing start and turn Apex + plot speed distributions at the two timepoints
    5. get the start of decelration + plot different metrics
"""


# get linearized track
(
    left_line,
    center_line,
    right_line,
    left_to_track,
    center_to_track,
    right_to_track,
    control_points,
) = track.extract_track_from_image(
    points_spacing=1, restrict_extremities=False, apply_extra_spacing=False,
)


# %%
"""
    Load, trim and filter data to get just nice clean turns
"""
# ----------------------------- load data - SLOW ----------------------------- #

# load bouts
bouts: List[Locomotion] = load_complete_bouts(keep=-1, window=5)


# ------------------- cut bouts to the frames withinthe ROI ------------------ #
ROI = ROIs_dict["T3"]

PARAMS = dict(
    T1=dict(),
    T2=dict(
        INITIAL_DECELERATION_CHECK=-0.5,
        DECELERATION_THRESHOLD=-1,
        INIT_SPEED_TH=40,
    ),
    T3=dict(
        INITIAL_DECELERATION_CHECK=-0.75,
        DECELERATION_THRESHOLD=-1,
        INIT_SPEED_TH=40,
    ),
)

INITIAL_DECELERATION_CHECK = PARAMS[ROI.name][
    "INITIAL_DECELERATION_CHECK"
]  # keep only trials with acc < this at start
DECELERATION_THRESHOLD = PARAMS[ROI.name][
    "DECELERATION_THRESHOLD"
]  # cm/s^2 -  the mice are delecerating when the longitudinal acceleration is below this value
INIT_SPEED_TH = PARAMS[ROI.name]["INIT_SPEED_TH"]

turns: List[Locomotion] = []
for bout in bouts:
    # get frames in ROI's gcoord range
    in_roi = np.where((bout.gcoord > ROI.g_0) & (bout.gcoord < ROI.g_1))[0]
    if np.any(np.diff(in_roi) > 1):
        raise ValueError("The mouse entered and exited the ROI, nicht good")

    # trim the locomotion bout
    turn = bout @ in_roi
    turn.gcoord = turn.gcoord[in_roi]

    # check that at start point the mouse isn't alrady decelerating
    at_start_index = np.where(turn.gcoord >= ROI.turn_g_start)[0][0]
    if (
        turn.body.longitudinal_acceleration[at_start_index]
        < INITIAL_DECELERATION_CHECK
    ):
        logger.debug("Discarded because mouse accelerating at start")
        continue

    # check that initial speed is sufficiently high
    if turn.body.speed[at_start_index] < INIT_SPEED_TH:
        logger.debug("Initial speed too low")
        continue

    # check that nowhere the mouse stops
    if np.any(turn.body.speed < 10):
        logger.debug("Discarded because mouse slows down too much")
        continue

    # check that acceleration dips below threshold before apex
    at_apex_index = np.where(turn.gcoord >= ROI.turn_g_apex)[0][0]
    try:
        slow_frame_idx = np.where(
            turn.body.longitudinal_acceleration < DECELERATION_THRESHOLD
        )[0][0]
    except IndexError:
        logger.debug("Discarded because couldnt find decelartion start time")
        continue
    if slow_frame_idx < at_start_index + 5 or slow_frame_idx > at_apex_index:
        logger.debug(
            "Discarded because decelartion start time is out of bounds"
        )
        continue

    turns.append(turn)
logger.info(f"Kept {len(turns)} turns")


# snapshots stores
start = AtPoint(indigo_dark, "start", G=ROI.turn_g_start)
apex = AtPoint(light_blue, "apex", G=ROI.turn_g_apex)
slowing = AtPoint(
    name="slowing", color=blue_darker
)  # when they start slowing down

peak_speed = []  # peak speed between start and apex
at_peak_speed = []

for n, turn in enumerate(turns):
    # start/apex
    for point in (start, apex):
        at_G_index = np.where(turn.gcoord >= point.G)[0][0]
        point.add(turn, at_G_index)

    # peak speed
    slowing_trace = turn @ np.arange(start.frame_idx[n], apex.frame_idx[n] + 1)
    peak_speed.append(np.max(slowing_trace.body.speed))
    at_peak_speed.append(np.argmax(slowing_trace.body.speed))

    # start of decelaration
    slow_frame_idx = np.where(
        turn.body.longitudinal_acceleration < DECELERATION_THRESHOLD
    )[0][0]
    slowing.add(turn, slow_frame_idx, center_line=center_line)

# %%
# ---------------------- speeds distributions histograms --------------------- #
"""
    Get tracking @ start and apex of turn and draw tracking traces + speed/acceleration profiles
    and histograms of the speed at start and apex of turn.
"""


f = plt.figure(figsize=(12, 12))
axes = f.subplot_mosaic(
    """
    CCAA
    BBAA
    BBAA
    DDAA
    DDAA
"""
)


draw.ROI(ROI.name, ax=axes["A"], set_ax=True)

# mark start/apex
for point in (start, apex):
    draw.gliphs.Dot(
        [loc.body.x for loc in point.locomotions],
        [loc.body.y for loc in point.locomotions],
        color=point.color,
        ax=axes["A"],
        zorder=500,
    )


for n, turn in enumerate(turns):
    _ = draw.Tracking(turn.body.x, turn.body.y, ax=axes["A"])

    # highlight trace from start -> end
    slowing_trace = turn @ np.arange(start.frame_idx[n], apex.frame_idx[n] + 1)
    slowing_trace.gcoord = turn.gcoord[
        start.frame_idx[n] : apex.frame_idx[n] + 1
    ]

    draw.Tracking(
        slowing_trace.body.x,
        slowing_trace.body.y,
        lw=3,
        ax=axes["A"],
        color=[0.2, 0.2, 0.2],
    )

    # plot speed and acceleration traces
    draw.Tracking(
        slowing_trace.gcoord,
        slowing_trace.body.speed,
        color=colors.speed,
        ax=axes["B"],
    )
    draw.Tracking(
        slowing_trace.gcoord,
        slowing_trace.body.longitudinal_acceleration,
        color=colors.acceleration,
        ax=axes["D"],
    )

    # get peak speed
    draw.Tracking.scatter(
        slowing_trace.body.x[at_peak_speed[n]],
        slowing_trace.body.y[at_peak_speed[n]],
        color=colors.speed,
        ax=axes["A"],
        zorder=250,
    )


# plot speeds histograms and mark means
draw.Hist(start.speed, color=start.color, ax=axes["C"], bins=6)
draw.Hist(apex.speed, color=apex.color, ax=axes["C"], bins=6)
for point in (start, apex):
    axes["C"].axvline(
        point.speed.mean(),
        lw=2,
        ls="--",
        zorder=400,
        color=point.color,
        alpha=0.8,
    )

axes["C"].set(ylabel="count", xlabel="speed (cm/s)", xlim=[10, 120])
_ = axes["B"].set(
    title="SPEED", xlabel="distance along track (norm)", ylabel="speed (cm/s)"
)
_ = axes["D"].set(
    title="ACCELERATION",
    xlabel="distance along track (norm)",
    ylabel="acceleration (cm/s**2)",
)

axes["D"].axhline(0, lw=2, color="k")
axes["D"].axhline(DECELERATION_THRESHOLD, lw=1, ls=":", zorder=300, color="k")

f.tight_layout()

# %%
# --------------------------- start of deceleration -------------------------- #
"""
    Analysis of when and where mice start decelarating based on their current speed
"""


f = plt.figure(figsize=(16, 12))
axes = f.subplot_mosaic(
    """
    AAACC
    AAADD
    AAAGG
"""
)


draw.ROI(ROI.name, set_ax=True, ax=axes["A"])
for n, turn in enumerate(turns):
    _ = draw.Tracking(turn.body.x, turn.body.y, alpha=1, lw=0.2, ax=axes["A"])
    # draw tracking traces colored by longitudinal accelration and normal
    draw.Tracking.scatter(
        turn.body.x[start.frame_idx[n] : apex.frame_idx[n]],
        turn.body.y[start.frame_idx[n] : apex.frame_idx[n]],
        c=turn.body.longitudinal_acceleration[
            start.frame_idx[n] : apex.frame_idx[n]
        ],
        cmap="bwr",
        vmin=-4,
        vmax=4,
        ax=axes["A"],
        zorder=200,
        lw=0.2,
        ec="k",
    )

# mark the position of the mice when they slow down
_ = draw.Tracking.scatter(
    slowing.x,
    slowing.y,
    c="k",
    zorder=300,
    ec="k",
    s=125,
    alpha=1,
    ax=axes["A"],
)


# plot distance along the track when the mice stop slowing down vs initial speed
_ = axes["C"].scatter(
    slowing.track_distance, slowing.speed, color=colors.velocity, s=100
)
sns.regplot(
    x=slowing.track_distance,
    y=slowing.speed,
    scatter=False,
    ax=axes["C"],
    robust=False,
    color=pink,
)

# # plot accelerationt traces from start of slowing down to apex
peak_decels = []
for n, turn in enumerate(turns):
    # get start -> apex trace
    slowing_trace = turn @ np.arange(start.frame_idx[n], apex.frame_idx[n])
    slowing_trace.gcoord = turn.gcoord[start.frame_idx[n] : apex.frame_idx[n]]

    # get position along track during start -> apex
    slowing_trace2center = TCS.path_to_track_coordinates_system(
        center_line, slowing_trace.body
    )

    # get peak deceleration
    peak_decel = np.min(slowing_trace.body.longitudinal_acceleration)
    peak_decel_frame = np.argmin(slowing_trace.body.longitudinal_acceleration)
    peak_decels.append(peak_decel)

# plot peak deceleration vs initial speed and peak deceleration histogram
axes["D"].scatter(peak_decels, start.speed, color="k", s=100)
sns.regplot(
    x=peak_decels,
    y=start.speed,
    scatter=False,
    ax=axes["D"],
    robust=False,
    color=pink,
)

draw.Hist(peak_decels, bins=10, color="k", ax=axes["G"])

# style axes
_ = axes["C"].set(
    title="deceleration position vs speed",
    xlabel="distance along track (norm)",
    ylabel="speed at slowing start (cm/s)",
    xlim=[
        np.min(slowing.track_distance) - 5,
        np.max(slowing.track_distance) + 5,
    ],
)
_ = axes["D"].set(
    title="deceleration peak vs speed",
    xlabel="peak deceleration (cm/s^2)",
    ylabel="speed at slowing start (cm/s)",
    xlim=[-4, -0.5],
)
_ = axes["G"].set(
    title="Peak decelerations",
    xlabel="peak deceleratiuon (cm/s^2)",
    ylabel="count",
    xlim=[-4, -0.5],
)


f.tight_layout()


# %%
