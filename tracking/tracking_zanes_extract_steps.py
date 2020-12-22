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
from vedo.colors import colorMap

from fcutils.maths.utils import rolling_mean

from fcutils.plotting.utils import (
    set_figure_subplots_aspect,
    clean_axes,
    save_figure,
)
from fcutils.plotting.plot_elements import vline_to_curve

import sys

sys.path.append("./")

from tracking._utils import draw_paws_steps, draw_mouse, mark_steps
from tracking.gait import (
    get_steps,
    stride_from_position,
    print_steps_summary,
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

starts = [  # how many frames before nice trot is reached
    50,
    60,
    55,
    50,
    40,
    40,
    40,
    40,
    55,
    55,
    55,
    55,
    50,
    45,
    48,
    55,
    57,
    55,
]

files = [f for f in folder.glob("*.h5") if f.name in turners]
print(f"Found {len(files)}/{len(turners)} files")


# ---------------------------------------------------------------------------- #
#                                    params                                    #
# ---------------------------------------------------------------------------- #

step_speed_th = 25  # cm / s
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
    f = plt.figure(constrained_layout=True, figsize=(20, 12))
    gs = f.add_gridspec(3, 5)

    tracking_ax = f.add_subplot(gs[:, 0])
    tracking_ax.axis("equal")
    paws_ax = f.add_subplot(gs[0, 1:3])
    cum_speeds_ax = f.add_subplot(gs[1, 1:3])
    ori_ax = f.add_subplot(gs[2, 1:3])
    turn_ax = f.add_subplot(gs[0, 3])

    x, y = 5, 45  # axes lims for summary plot
    turn_ax.set(  # summary ax
        title="stride difference vs turn angle",
        xlim=[-x, x],
        xticks=[-x, 0, x],
        xticklabels=[f"R>L\n{-x}", "R=L\n0", f"R<L\n{x}"],
        xlabel="(L-R) stride-delta\m(cm)",
        ylim=[-y, y],
        yticks=[-y, 0, y],
        yticklabels=[f"turn\nleft\n{-y}", "no\nturn\n0", f"turn\nright\n{y}"],
        ylabel="(end-start) angle-delta\n(deg)",
    )
    turn_ax.axvline(0, ls="--", color=[0.5, 0.5, 0.5])
    turn_ax.axhline(0, ls="--", color=[0.5, 0.5, 0.5])

    tracking_ax.axis("off")
    paws_ax.set(title="paw speed", ylabel="speed\ncm/s", xlabel="Time\nframes")
    cum_speeds_ax.set(
        title="Dist.travelled", ylabel="Distance\ncm", xlabel="Time\nframes"
    )
    ori_ax.set(
        title="orientation", ylabel="ang\ndegrees", xlabel="Time\nframes"
    )

    set_figure_subplots_aspect(wspace=0.6, hspace=0.4)
    clean_axes(f)

    return f, tracking_ax, paws_ax, cum_speeds_ax, ori_ax, turn_ax


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
    if not np.any(np.isnan(a)):
        return np.degrees(np.unwrap(np.radians(a)))
    else:
        # unwrap only non-nan
        idx = np.where(np.isnan(a))[0][-1] + 1

        out = np.zeros_like(a)
        out[idx:] = np.degrees(np.unwrap(np.radians(a[idx:])))
        out[:idx] = np.nan
        return out


for runn, (_file, start) in enumerate(zip(files, starts)):
    if runn != 1:
        continue

    # load tracking data
    tracking = pd.read_hdf(_file, key="hdf")

    # make figure
    f, tracking_ax, paws_ax, cum_speeds_ax, ori_ax, turn_ax = make_figure()

    # --------------------------------- get steps -------------------------------- #
    LH_speed = t(tracking[f"LH_speed"]) * fps
    LF_speed = t(tracking[f"LF_speed"]) * fps
    RH_speed = t(tracking[f"RH_speed"]) * fps
    RF_speed = t(tracking[f"RF_speed"]) * fps

    (
        LH_steps,
        RF_steps,
        LF_steps,
        RH_steps,
        diagonal_steps,
        diag_data,
        step_starts,
    ) = get_steps(
        LH_speed, LF_speed, RH_speed, RF_speed, step_speed_th, start=start
    )

    # -------------------------------- draw mouse -------------------------------- #
    draw_mouse(tracking_ax, tracking, step_starts)
    draw_paws_steps(paw_colors, tracking_ax, tracking, step_starts, start)

    # -------------------------------- other plots ------------------------------- #
    # Plot paw speeds
    for n, (paw, color) in enumerate(paw_colors.items()):
        if paw in ("LF", "RH"):
            alpha = 0.2
        else:
            alpha = 1
        spd = t(tracking[f"{paw}_speed"]) * fps

        paws_ax.plot(spd, color=color, lw=2, alpha=alpha, label=paw)
        cum_speeds_ax.plot(
            np.cumsum(spd), color=color, lw=2, alpha=alpha, label=paw
        )
    paws_ax.legend()
    cum_speeds_ax.legend()
    paws_ax.axhline(step_speed_th, lw=1, ls=":", color="k", zorder=-1)
    paws_ax.axhline(0, lw=2, color="k", zorder=1)

    # Plot orientation
    orientation = t(r(tracking["body_whole_bone_orientation"]), scale=False)
    ori_ax.plot(
        orientation, label="body angle", color=blue_grey_darker, lw=4,
    )
    ori_ax.legend()

    # draw steps times
    for n, (paw, steps) in enumerate(
        zip(("RF", "LH", "LF", "RH"), (RF_steps, LH_steps, LF_steps, RH_steps))
    ):
        mark_steps(
            paws_ax,
            steps.starts,
            steps.ends,
            -10 * (n + 1),
            paw,
            5,
            color=paw_colors[paw],
            alpha=0.9,
            zorder=-1,
            lw=2,
        )

    # draw diagonal steps
    for ax, offest, scale, noise in zip(
        (paws_ax, cum_speeds_ax, ori_ax),
        (-80, -0.2, 0),
        (5, 0.05, 0.5),
        (5, 20, 1),
    ):
        mark_steps(
            ax,
            diagonal_steps.starts,
            diagonal_steps.ends,
            offest,
            "Diag.",
            scale,
            noise=noise,
            alpha=1,
            zorder=-1,
            lw=2,
        )

    # ------------------------------ stride vs angle ----------------------------- #
    # get stride length vs turn angle
    summary = dict(
        number=[],
        stride_delta=[],
        angle_delta=[],
        side=[],
        start=[],
        end=[],
        pearsonr=[],
    )
    for n, step in diag_data.items():
        # stride delta
        r_stride = stride_from_position(
            t(tracking[f"RF_x"]),
            t(tracking[f"RF_y"]),
            step["fore_start"],
            step["fore_end"],
        )
        l_stride = stride_from_position(
            t(tracking[f"LH_x"]),
            t(tracking[f"LH_y"]),
            step["hind_start"],
            step["hind_end"],
        )
        summary["stride_delta"].append(l_stride - r_stride)

        # angle delta
        turn = orientation[step["end"]] - orientation[step["start"]]
        summary["angle_delta"].append(turn)

        # more info
        summary["number"].append(n)
        summary["side"].append(step["side"])
        summary["start"].append(step["start"])
        summary["end"].append(step["end"])
        summary["pearsonr"].append(step["pearsonr"])

        # add to orientation and speeds plots
        color = colorMap(n, name="inferno", vmin=0, vmax=len(diag_data))
        for frm in [step["start"], step["end"]]:
            vline_to_curve(
                ori_ax,
                frm,
                np.arange(len(orientation)),
                orientation,
                dot=True,
                line_kwargs=dict(lw=1, color="k"),
                scatter_kwargs=dict(
                    lw=2,
                    edgecolors="k",
                    s=80,
                    color=color,
                    zorder=200,
                    alpha=1,
                ),
            )

            vline_to_curve(
                cum_speeds_ax,
                frm,
                np.arange(len(orientation)),
                np.cumsum(t(tracking[f"LH_speed"]) * fps),
                dot=True,
                line_kwargs=dict(lw=1, color=paw_colors["LH"]),
                scatter_kwargs=dict(
                    lw=2,
                    edgecolors="k",
                    s=80,
                    color=color,
                    zorder=200,
                    alpha=1,
                ),
            )

    # plot stride length vs turn angle
    turn_ax.scatter(
        summary["stride_delta"],
        summary["angle_delta"],
        s=80,
        c=[1 if s == "R" else 0 for s in summary["side"]],
        zorder=10,
        lw=1,
        edgecolors=[0.4, 0.4, 0.4],
        cmap="bwr",
    )

    print_steps_summary(summary)

    # cache steps summary
    fname = _file.name.split(".h5")[0]
    expval = folder.parent.parent / "experimental_validation"
    pd.DataFrame(summary).to_hdf(
        expval / (fname + "_steps_cache.h5"), key="hdf"
    )

    # save figure
    f.tight_layout()
    save_figure(
        f, expval / "plots" / fname,
    )

plt.show()

# %%
