# %%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from myterial import (
    salmon_darker,
    indigo_darker,
    indigo,
    salmon,
    blue_grey_darker,
)

from fcutils.maths.utils import rolling_mean, derivative

from fcutils.plotting.utils import set_figure_subplots_aspect, clean_axes

from tracking._utils import line, point, draw_mouse

# %%

# ---------------------------------------------------------------------------- #
#                                   get data                                   #
# ---------------------------------------------------------------------------- #

folder = Path(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/control/behav_data/Zane"
)


turners = [
    "ZM_201002_ZM012_escconcat_0.h5",
    "ZM_201002_ZM011_escconcat_5.h5",
    "ZM_201002_ZM012_escconcat_6.h5",
    "ZM_201002_ZM012_escconcat_8.h5",
    "ZM_201002_ZM012_escconcat_10.h5",
    "ZM_201002_ZM012_escconcat_12.h5",
    "ZM_201002_ZM012_escconcat_15.h5",
    "ZM_201002_ZM014_escconcat_14.h5",
    "ZM_201002_ZM015_escconcat_4.h5",
    "ZM_201002_ZM015_escconcat_8.h5",
    "ZM_201002_ZM015_escconcat_12.h5",
    "ZM_201003_ZM017_escconcat_11.h5",
    "ZM_201003_ZM018_escconcat_0.h5",
    "ZM_201003_ZM018_escconcat_6.h5",
    "ZM_201003_ZM018_escconcat_8.h5",
    "ZM_201003_ZM018_escconcat_9.h5",
    "ZM_201003_ZM020_escconcat_2.h5",
    "ZM_201003_ZM020_escconcat_5.h5",
    "ZM_201003_ZM020_escconcat_8.h5",
    "ZM_201003_ZM020_escconcat_19.h5",
    "ZM_201003_ZM020_escconcat_23.h5",
]

starts = [
    55,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

files = [f for f in folder.glob("*.h5") if f.name in turners]
print(f"Found {len(files)}/{len(turners)} files")


# ---------------------------------------------------------------------------- #
#                                    params                                    #
# ---------------------------------------------------------------------------- #

step_speed_th = 0.5

paws = ("LF", "RF", "LH", "RH")
paw_colors = {
    "LF": indigo_darker,
    "RF": salmon_darker,
    "LH": salmon,
    "RH": indigo,
}

fps = 60
cm_per_px = 1 / 30.8
# %%

# ------------------------------- useful funcs ------------------------------- #


def make_figure():
    f = plt.figure(constrained_layout=True, figsize=(26, 12))
    gs = f.add_gridspec(3, 5)

    tracking_ax = f.add_subplot(gs[:, 0])
    tracking_ax.axis("equal")
    paws_ax = f.add_subplot(gs[0, 1:3])
    bones_ax = f.add_subplot(gs[1, 1:3])
    ori_ax = f.add_subplot(gs[2, 1:3])
    turn_ax = f.add_subplot(gs[0, 3])

    turn_ax.set(
        title="stride difference vs turn angle",
        xlim=[-30, 30],
        xticks=[-30, 0, 30],
        xticklabels=["R>L", "R+L", "R<L"],
        xlabel="left stride - right stride",
        yticks=[-0.3, 0, 0.3],
        yticklabels=["turn\nright", "no\nturn", "turn\nleft"],
    )
    turn_ax.axvline(0, ls="--", color=[0.5, 0.5, 0.5])
    turn_ax.axhline(0, ls="--", color=[0.5, 0.5, 0.5])

    tracking_ax.axis("off")
    paws_ax.set(title="paw speed", ylabel="speed\ncm/s", xlabel="Time\nframes")
    bones_ax.set(
        title="Side length", ylabel="length\ncm", xlabel="Time\nframes"
    )
    ori_ax.set(
        title="Orientation", ylabel="angle\ndegrees", xlabel="Time\nframes"
    )

    set_figure_subplots_aspect(wspace=0.4, hspace=0.4)
    clean_axes(f)

    return f, tracking_ax, paws_ax, bones_ax, ori_ax, turn_ax


def t(d):
    """
        Transform data by smoothing and
        going  from px to cm
    """
    try:
        d = d.values
    except Exception:
        pass

    return rolling_mean(d[start:], 3) * cm_per_px


def r(a):
    """
        unwrap circular data
    """
    return np.degrees(np.unwrap(np.radians(a)))


def get_steps(speed):
    """
        Given the speed of a paw, find when the swing
        phase starts and ends
    """
    is_swing = np.zeros_like(speed)
    is_swing[speed > step_speed_th] = 1
    first_zero = np.where(is_swing == 0)[0][0]
    is_swing[:first_zero] = 0  # make sure that is starts with swing phase OFF

    starts = np.where(derivative(is_swing) > 0)[0]
    ends = np.where(derivative(is_swing) < 0)[0]
    return starts, ends


# %%

for runn, (f, start) in enumerate(zip(files, starts)):
    # load tracking data
    tracking = pd.read_hdf(f, key="hdf")

    # make figure
    f, tracking_ax, paws_ax, bones_ax, ori_ax, turn_ax = make_figure()

    # --------------------------------- get steps -------------------------------- #

    frames = get_steps(t(tracking[f"LH_speed"]))[0] + start

    # mark steps
    for fm in frames:
        paws_ax.axvline(fm - start, lw=1, color=[0.2, 0.2, 0.2], zorder=-1)

    # -------------------------------- draw mouse -------------------------------- #
    draw_mouse(tracking_ax, tracking, frames)

    # Plot paws
    for paw, color in paw_colors.items():
        point(paw, tracking_ax, tracking, frames, zorder=1, color=color, s=50)

    # plot paw lines
    line(
        "LH",
        "RF",
        tracking_ax,
        tracking,
        frames,
        color=salmon,
        lw=2,
        zorder=2,
    )
    line(
        "RH",
        "LF",
        tracking_ax,
        tracking,
        frames,
        color=indigo,
        lw=2,
        zorder=2,
    )

    # -------------------------------- other plots ------------------------------- #
    # Plot paw speeds
    for n, (paw, color) in enumerate(paw_colors.items()):
        if "fore" in paw:
            continue
        y = t(tracking[f"{paw}_speed"]) * cm_per_px * fps

        paws_ax.plot(y, color=color, lw=3, alpha=0.8, label=paw)
    paws_ax.legend()
    paws_ax.axhline(step_speed_th, lw=1, ls=":", color="k", zorder=-1)

    # Plot bone lengths
    bones_ax.plot(
        t(tracking["left_bone_length"]),
        color=salmon_darker,
        lw=3,
        label="LF-LH distance",
    )
    bones_ax.plot(
        t(tracking["right_bone_length"]),
        color=indigo_darker,
        lw=3,
        label="RF-RH distance",
    )
    bones_ax.legend()

    # Plot orientation
    orientation = t(r(tracking["body_lower_bone_orientation"]))
    ori_ax.plot(
        orientation, label="body angle", color=blue_grey_darker, lw=4,
    )
    ori_ax.legend()

    # get stride length vs turn angle
    summary = dict(L_stride=[], R_stride=[], turn_angle=[],)

    for n, paw in enumerate(("R", "L")):
        s1s, s2s = get_steps(t(tracking[f"{paw}H_speed"]))
        for s1, s2 in zip(s1s, s2s):
            y = -(n * 0.1) - 0.1

            for ax in (paws_ax, bones_ax, ori_ax):
                ax.axvspan(
                    s1,
                    s2,
                    color=paw_colors[paw + "H"],
                    alpha=0.15,
                    zorder=-1,
                    lw=0,
                )

            distance = np.cumsum(t(tracking[f"{paw}H_speed"])[s1:s2])
            summary[f"{paw}_stride"].append(np.sum(distance))

            if paw == "R":
                summary["turn_angle"].append(orientation[s2] - orientation[s1])

    # plot stride length vs turn angle
    for left, right, angle in zip(*summary.values()):
        turn_ax.scatter(
            (left - right), angle, s=80, color=[0.2, 0.2, 0.2], zorder=10
        )

    break

plt.show()
# %%

# %%
