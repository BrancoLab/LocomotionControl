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

from rich.table import Table
from rich import print
from rich.box import SIMPLE_HEAD

from fcutils.maths.utils import rolling_mean

from fcutils.plotting.utils import set_figure_subplots_aspect, clean_axes

from tracking._utils import line, point, draw_mouse, mark_steps
from tracking.gait import (
    get_paw_steps_times,
    get_diagonal_steps,
    stride_from_position,
)

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

step_speed_th = 20  # cm / s
fps = 60
cm_per_px = 1 / 30.8


paws = ("LF", "RF", "LH", "RH")
paw_colors = {
    "LF": indigo_darker,
    "RF": salmon_darker,
    "LH": salmon,
    "RH": indigo,
}


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

    x, y = 3, 0.75  # axes lims for summary plot
    turn_ax.set(  # summary ax
        title="stride difference vs turn angle",
        xlim=[-x, x],
        xticks=[-x, 0, x],
        xticklabels=["R>L", "R=L", "R<L"],
        xlabel="(L-R) stride delta",
        ylim=[-y, y],
        yticks=[-0.3, 0, 0.3],
        yticklabels=["turn\nright", "no\nturn", "turn\nleft"],
        ylabel="(end-start) angle delta",
    )
    turn_ax.axvline(0, ls="--", color=[0.5, 0.5, 0.5])
    turn_ax.axhline(0, ls="--", color=[0.5, 0.5, 0.5])

    tracking_ax.axis("off")
    paws_ax.set(title="paw speed", ylabel="speed\ncm/s", xlabel="Time\nframes")
    bones_ax.set(
        title="Side length", ylabel="length\ncm", xlabel="Time\nframes"
    )
    ori_ax.set(
        title="orientation", ylabel="ang\ndegrees", xlabel="Time\nframes"
    )

    set_figure_subplots_aspect(wspace=0.4, hspace=0.4)
    clean_axes(f)

    return f, tracking_ax, paws_ax, bones_ax, ori_ax, turn_ax


def t(d, scale=True):
    """
        Transform data by smoothing and
        going  from px to cm
    """
    try:
        d = d.values
    except Exception:
        pass

    d = rolling_mean(d[start:], 2)

    if scale:
        return d * cm_per_px
    else:
        return d


def r(a):
    """
        unwrap circular data
    """
    return np.degrees(np.unwrap(np.radians(a)))


def steps_summary(diag_steps_data, summary):
    tb = Table(
        header_style="bold green",
        show_lines=True,
        expand=False,
        box=SIMPLE_HEAD,
    )
    tb.add_column("#", style="dim")
    tb.add_column("start", justify="center")
    tb.add_column("end", justify="center")
    tb.add_column("dur.", justify="center")
    tb.add_column("stride delta", justify="right")
    tb.add_column("angle delta", justify="right")

    for n, data in diag_steps_data.items():
        tb.add_row(
            str(n),
            str(data["leading_start"]),
            str(data["trailing_end"]),
            str(data["trailing_end"] - data["leading_start"]),
            f"{summary['stride_delta'][n]:.3f}",
            f"{summary['angle_delta'][n]:.3f}",
        )

    print("\n", tb)


# %%

for runn, (f, start) in enumerate(zip(files, starts)):
    # load tracking data
    tracking = pd.read_hdf(f, key="hdf")

    # make figure
    f, tracking_ax, paws_ax, bones_ax, ori_ax, turn_ax = make_figure()

    # --------------------------------- get steps -------------------------------- #
    L_steps = get_paw_steps_times(
        t(tracking[f"LH_speed"]) * fps, step_speed_th
    )
    R_steps = get_paw_steps_times(
        t(tracking[f"RF_speed"]) * fps, step_speed_th
    )
    diagonal_steps, first_step_side, diag_steps_data = get_diagonal_steps(
        L_steps, R_steps
    )

    step_starts = (
        np.array(diagonal_steps.starts) + start
    )  # to mark the start of each L-R step sequence

    # mark steps
    # for fm in step_starts:
    #     paws_ax.axvline(fm - start, lw=1, color=[0.2, 0.2, 0.2], zorder=-1)

    # -------------------------------- draw mouse -------------------------------- #
    draw_mouse(tracking_ax, tracking, step_starts)

    # Plot paws
    for paw, color in paw_colors.items():
        point(
            paw,
            tracking_ax,
            tracking,
            step_starts,
            zorder=1,
            color=color,
            s=50,
        )

    # plot paw lines
    line(
        "LH",
        "RF",
        tracking_ax,
        tracking,
        step_starts,
        color=salmon,
        lw=2,
        zorder=2,
    )
    line(
        "RH",
        "LF",
        tracking_ax,
        tracking,
        step_starts,
        color=indigo,
        lw=2,
        zorder=2,
    )

    # -------------------------------- other plots ------------------------------- #
    # Plot paw speeds
    for n, (paw, color) in enumerate(paw_colors.items()):
        if "F" in paw:
            alpha = 0.2
        else:
            alpha = 1
        y = t(tracking[f"{paw}_speed"]) * fps

        paws_ax.plot(y, color=color, lw=2, alpha=alpha, label=paw)
    paws_ax.legend()
    paws_ax.axhline(step_speed_th, lw=1, ls=":", color="k", zorder=-1)
    paws_ax.axhline(0, lw=2, color="k", zorder=1)

    # Plot bone lengths
    bones_ax.plot(
        t(tracking["left_bone_length"])
        / t(tracking["body_whole_bone_length"]),
        color=salmon_darker,
        lw=3,
        label="LF-LH distance",
    )
    bones_ax.plot(
        t(tracking["right_bone_length"])
        / t(tracking["body_whole_bone_length"]),
        color=indigo_darker,
        lw=3,
        label="RF-RH distance",
    )
    bones_ax.legend()

    # Plot orientation
    orientation = t(r(tracking["body_lower_bone_orientation"]), scale=False)
    ori_ax.plot(
        orientation, label="body angle", color=blue_grey_darker, lw=4,
    )
    ori_ax.legend()

    # draw steps times
    for n, (paw, steps) in enumerate(zip(("R", "L"), (R_steps, L_steps))):
        for ax, offest, scale in zip(
            (paws_ax, bones_ax, ori_ax), (-20, -0.2, 385), (5, 0.05, 0.5)
        ):
            y = (n * scale) + offest
            mark_steps(
                ax,
                steps.starts,
                steps.ends,
                y,
                paw,
                scale,
                color=paw_colors[paw + "H"],
                alpha=0.9,
                zorder=-1,
                lw=2,
            )
    mark_steps(
        paws_ax,
        diagonal_steps.starts,
        diagonal_steps.ends,
        -60,
        "Diag.",
        5,
        color=blue_grey_darker,
        noise=10,
        alpha=1,
        zorder=-1,
        lw=2,
    )

    # ------------------------------ stride vs angle ----------------------------- #
    # get stride length vs turn angle
    summary = dict(stride_delta=[], angle_delta=[])
    for n, step in diag_steps_data.items():

        if first_step_side == "right":
            r_start, r_end = step["leading_start"], step["leading_end"]
            l_start, l_end = step["trailing_start"], step["trailing_end"]
        else:
            r_start, r_end = step["trailing_start"], step["trailing_end"]
            l_start, l_end = step["leading_start"], step["leading_end"]

        # stride delta
        r_stride = stride_from_position(
            t(tracking[f"RH_x"]), t(tracking[f"RH_y"]), r_start, r_end
        )
        l_stride = stride_from_position(
            t(tracking[f"LH_x"]), t(tracking[f"LH_y"]), l_start, l_end
        )
        summary["stride_delta"].append(l_stride - r_stride)

        # angle delta
        summary["angle_delta"].append(
            orientation[step["trailing_end"]]
            - orientation[step["leading_start"]]
        )

    # plot stride length vs turn angle
    turn_ax.scatter(
        summary["stride_delta"],
        summary["angle_delta"],
        s=80,
        c=np.arange(len(summary["angle_delta"])),
        zorder=10,
        lw=1,
        edgecolors=[0.4, 0.4, 0.4],
        cmap="Reds",
    )

    steps_summary(diag_steps_data, summary)
    break

plt.show()
# %%

# %%
