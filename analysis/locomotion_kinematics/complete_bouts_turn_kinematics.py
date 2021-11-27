# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
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


"""
    Analysis of the turn kinematics with data from complete bouts but looking only at single turns.


    1. define AtPoint calss to collect kinematics at same time point across locomotion instances
    2. load data (slow) + trim for selected ROI
    3. draw tracking data
    4. get AtPoint for ROI crossing start and turn Apex + plot speed distributions at the two timepoints
    5. get the start of decelration + plot different metrics
"""


@dataclass
class AtPoint:
    """
        Class used to collect kinematics data across different Locomotion traces, for each
        trace it stores the locomotion data for a single time point (i.e. when the mouse is at
        a selected point in the arena). Useful to e.g. compare kinematics everytime the mice are
        at the apex of a turn.
    """

    color: str
    name: str
    frame_idx: list = field(
        default_factory=list
    )  # index of the selected frame of each locomotion
    G: Optional[float] = None  # g_coord value
    locomotions: list = field(default_factory=list)
    track_distance: list = field(
        default_factory=list
    )  # distance along linearized track at frame
    G_distance: list = field(default_factory=list)  # gcoord at selected frame

    @property
    def x(self) -> np.ndarray:
        return np.array([loc.body.x for loc in self.locomotions])

    @property
    def y(self) -> np.ndarray:
        return np.array([loc.body.y for loc in self.locomotions])

    @property
    def speed(self) -> np.ndarray:
        return np.array([loc.body.speed for loc in self.locomotions])


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
# ----------------------------- load data - SLOW ----------------------------- #

# load bouts
bouts: List[Locomotion] = load_complete_bouts(keep=None, window=5)


# ------------------- cut bouts to the frames withinthe ROI ------------------ #
ROI = ROIs_dict["T2"]

INITIAL_DECELERATION_CHECK = -0.5  # keep only trials with acc < this at start
DECELERATION_THRESHOLD = (
    -1
)  # cm/s^2 -  the mice are delecerating when the longitudinal acceleration is below this value

START_G = 0.22
APEX_G = 0.34


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
    at_start_index = np.where(turn.gcoord >= START_G)[0][0]
    if (
        turn.body.longitudinal_acceleration[at_start_index]
        < INITIAL_DECELERATION_CHECK
    ):
        continue

    # check that nowhere the mouse stops
    if np.any(turn.body.speed < 10):
        continue

    # check that acceleration dips below threshold before apex
    at_apex_index = np.where(turn.gcoord >= APEX_G)[0][0]
    try:
        slow_frame_idx = np.where(
            turn.body.longitudinal_acceleration < DECELERATION_THRESHOLD
        )[0][0]
    except IndexError:
        continue
    if slow_frame_idx < at_start_index + 10 or slow_frame_idx > at_apex_index:
        continue

    turns.append(turn)
logger.info(f"Kept {len(turns)} turns")


# %%
# ---------------------- speeds distributions histograms --------------------- #
"""
    Draw histograms of speed values at ROI enter and apex of turn and traces for
    speed and acceleration.
"""

start = AtPoint(indigo_dark, "start", G=START_G)
apex = AtPoint(light_blue, "apex", G=APEX_G)

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

# get kinematics at turn start and apex (+ plot)
for turn in turns:
    for point in (start, apex):
        at_G_index = np.where(turn.gcoord >= point.G)[0][0]
        at_G = turn @ at_G_index
        point.locomotions.append(at_G)
        point.frame_idx.append(at_G_index)

        draw.gliphs.Dot(
            at_G.body.x,
            at_G.body.y,
            color=point.color,
            ax=axes["A"],
            zorder=500,
        )
    _ = draw.Tracking(turn.body.x, turn.body.y, ax=axes["A"])


peak_speed = []  # peak speed between start and apex
for n, turn in enumerate(turns):
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
    peak_speed.append(np.max(slowing_trace.body.speed))
    at_peak = np.argmax(slowing_trace.body.speed)
    draw.Tracking.scatter(
        slowing_trace.body.x[at_peak],
        slowing_trace.body.y[at_peak],
        color=colors.speed,
        ax=axes["A"],
        zorder=250,
    )


# plot speeds histograms and mark means
draw.Hist(peak_speed, color=start.color, ax=axes["C"], bins=6)
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


slowing = AtPoint(name="slowing", color=blue_darker)

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
    )

    # get when the mouse starts decelaring
    slow_frame_idx = np.where(
        turn.body.longitudinal_acceleration < DECELERATION_THRESHOLD
    )[0][0]
    at_slow = turn @ slow_frame_idx
    slowing.locomotions.append(at_slow)
    slowing.frame_idx.append(slow_frame_idx)
    slowing.track_distance.append(
        TCS.point_to_track_coordinates_system(
            center_line, (at_slow.body.x, at_slow.body.y)
        )[0]
    )
    slowing.G_distance.append(turn.gcoord[slow_frame_idx])

# mark the position of the mice when they slow down
_ = draw.Tracking.scatter(
    slowing.x,
    slowing.y,
    c="k",
    zorder=100,
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
    xlim=[55, 80],
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
