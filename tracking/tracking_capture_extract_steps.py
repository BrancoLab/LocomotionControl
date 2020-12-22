import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyinspect import install_traceback

install_traceback()

from fcutils.maths.geometry import calc_distance_from_point
from fcutils.maths.stimuli_detection import get_onset_offset
from fcutils.maths.utils import rolling_mean, derivative
from fcutils.plotting.utils import clean_axes, set_figure_subplots_aspect

from myterial import (
    salmon_darker,
    indigo_darker,
    indigo,
    salmon,
)
import sys

sys.path.append("./")

from tracking._utils import draw_mouse, point

"""
    Body parts
        SpineF -> cervical spine
        SpineM -> thoracic spine
        SpineL -> lumbar spine
        Arm -> near the wrist
        Shin -> near the ankle

"""

# params
fps = 300
DISTANCE_TH = 225  # any further than this and you're at the wall
SPEED_TH = 10  # cm/s, any slower and you're not walking
MIN_BOUT_DURATION = (
    1 * fps
)  # any shorter than this an it's not a locomotion bout

STEP_SPEED_TH = 20  # cm/s initial TH for step detection

# ----------------------------------- load ----------------------------------- #

# load files
data = pd.read_hdf(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/control/behav_data/capture_cleaned.h5",
    key="hdf",
)
# print(data.columns)
# data = data[:100000]  # TODO remove

# rename columns
data.columns = [
    "LF_x",
    "RF_x",
    "LH_x",
    "RH_x",
    "body_x",
    "upper_body_x",
    "lower_body_x",
    "LF_y",
    "RF_y",
    "LH_y",
    "RH_y",
    "body_y",
    "upper_body_y",
    "lower_body_y",
    "LF_speed",
    "RF_speed",
    "LH_speed",
    "RH_speed",
    "body_speed",
    "upper_body_speed",
    "lower_body_speed",
    "LF_direction_of_movement",
    "RF_direction_of_movement",
    "LH_direction_of_movement",
    "RH_direction_of_movement",
    "body_direction_of_movement",
    "upper_body_direction_of_movement",
    "lower_body_direction_of_movement",
    "LF_angular_velocity",
    "RF_angular_velocity",
    "LH_angular_velocity",
    "RH_angular_velocity",
    "body_angular_velocity",
    "upper_body_angular_velocity",
    "lower_body_angular_velocity",
    "body1_bone_length",
    "body2_bone_length",
    "body3_bone_length",
    "body1_bone_orientation",
    "body2_bone_orientation",
    "in_center",
]

# ------------------------------- center bouts ------------------------------- #

# get distance from center
center = np.array([60, 10])
center_distance = calc_distance_from_point(
    data[["body_x", "body_y"]].values, center
)

data["center_distance"] = center_distance
data["in_center"] = center_distance < DISTANCE_TH

# Get bouts starts/ends
sts, eds = get_onset_offset(data.center_distance.values, DISTANCE_TH)
bouts_starts = eds
bouts_ends = sts[1:]

# get speed
speed = (
    np.mean(
        data[["body_speed", "lower_body_speed", "upper_body_speed"]].values, 1
    )
    * fps
)
speed = rolling_mean(speed, int(0.05 * fps))

# keep only when rat is walking
locomotion_starts, locomotion_ends = [], []
for s, e in zip(bouts_starts, bouts_ends):
    walk_start, walk_end = get_onset_offset(speed[s:e], SPEED_TH, clean=True)

    for ws, we in zip(walk_start, walk_end):
        if we - ws > MIN_BOUT_DURATION:
            if np.any(speed[s + ws : s + we] < SPEED_TH):
                continue
            locomotion_starts.append(s + ws)
            locomotion_ends.append(s + we)


def angle(arr):
    """ unwrap and baseline angle """
    ang = np.degrees(np.unwrap(np.radians(arr)))
    return ang - ang[0]


# Keep only buots with some path curvature
bouts_summary = dict(start=[], end=[], duration=[], turn_angle=[])
for s, e in zip(locomotion_starts, locomotion_ends):
    turn = angle(data.body1_bone_orientation.values[s:e])

    bouts_summary["start"].append(int(s))
    bouts_summary["end"].append(int(e))
    bouts_summary["duration"].append(e - s)
    bouts_summary["turn_angle"].append(np.sum(np.abs(derivative(turn))))


# -------------------------- plot overall recording -------------------------- #
if False:  # TODO remov
    f, axarr = plt.subplots(ncols=2, nrows=2, figsize=(9, 9))
    axarr = axarr.flatten()

    for s, e in zip(locomotion_starts, locomotion_ends):
        axarr[0].scatter(
            data["body_x"][s], data["body_y"][s], color="k", s=30, zorder=2
        )
        axarr[0].scatter(
            data["body_x"][s:e].values[::2],
            data["body_y"][s:e].values[::2],
            c=derivative(angle(data.body1_bone_orientation.values[s:e]))[::2],
            s=10,
            cmap="bwr",
            edgecolors=[0.2, 0.2, 0.2],
            lw=0.2,
        )

        axarr[1].plot(speed[s:e], color="k", lw=1, alpha=0.4)

        axarr[2].hist(bouts_summary["turn_angle"])

    # clean axes
    axarr[0].axis("equal")
    axarr[0].set(
        title="tracking", ylabel="Y", xlabel="X", xticks=[], yticks=[]
    )
    axarr[1].set(title="Speed", ylabel="speed\ncm/s", xlabel="time (frames)")
    clean_axes(f)


# ---------------------------------------------------------------------------- #
#                               steps extraction                               #
# ---------------------------------------------------------------------------- #

# ----------------------------------- utils ---------------------------------- #


def make_figure():
    f = plt.figure(constrained_layout=True, figsize=(18, 10))
    gs = f.add_gridspec(2, 4)

    tracking_ax = f.add_subplot(gs[:, :2])
    tracking_ax.axis("equal")
    paws_ax = f.add_subplot(gs[0, 2:])

    tracking_ax.axis("off")
    paws_ax.set(title="paw speed", ylabel="speed\ncm/s", xlabel="Time\nframes")

    set_figure_subplots_aspect(wspace=0.6, hspace=0.4)
    clean_axes(f)

    return f, tracking_ax, paws_ax


def t(d, scale=True):
    """
        Transform data by smoothing and
        going  from px to cm
    """
    try:
        d = d.values
    except Exception:
        pass

    return rolling_mean(d, 60)


paw_colors = {
    "LF": indigo_darker,
    "RF": salmon_darker,
    "LH": salmon,
    "RH": indigo,
}


# --------------------------------- run bouts -------------------------------- #
bouts = pd.DataFrame(bouts_summary).sort_values("duration", 0)
for i, bout in bouts.iterrows():
    if i != 5:
        continue

    # get the speed of each limb
    LF_speed, LH_speed = data["LF_speed"] * fps, data["LH_speed"] * fps
    RF_speed, RH_speed = data["RF_speed"] * fps, data["RH_speed"] * fps

    f, tracking_ax, paws_ax = make_figure()

    # draw mouse
    s, e, dur = int(bout.start), int(bout.end), int(bout.duration)
    tracking = data[s:e]
    draw_frames = np.arange(dur)[::60]
    draw_mouse(
        tracking_ax,
        tracking,
        draw_frames,
        bps=("upper_body", "LF", "LH", "lower_body", "RH", "RF"),
    )

    for paw, color in paw_colors.items():
        point(
            paw,
            tracking_ax,
            tracking,
            draw_frames,
            zorder=1,
            color=color,
            s=50,
        )

    # plot paws speed
    for n, (paw, color) in enumerate(paw_colors.items()):
        if paw in ("LH", "RH"):
            alpha = 0.2
        else:
            alpha = 1
        spd = tracking[f"{paw}_speed"] * fps
        paws_ax.plot(t(spd), color=color, lw=2, alpha=alpha, label=paw)
    paws_ax.legend()
    paws_ax.axhline(STEP_SPEED_TH, lw=1, ls=":", color="k", zorder=-1)
    paws_ax.axhline(0, lw=2, color="k", zorder=1)

    break

    # (
    #     LH_steps,
    #     RF_steps,
    #     LF_steps,
    #     RH_steps,
    #     diagonal_steps,
    #     diag_data,
    #     step_starts,
    # ) = get_steps (
    #     LH_speed,
    #     LF_speed,
    #     RH_speed,
    #     RF_speed
    #     STEP_SPEED_TH
    #     )


plt.show()
