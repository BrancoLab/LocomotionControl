from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import sys
import numpy as np

from fcutils.plotting.utils import (
    set_figure_subplots_aspect,
    clean_axes,
)
from fcutils.maths.utils import rolling_mean
from myterial import (
    salmon_darker,
    indigo_darker,
    indigo,
    salmon,
    blue_grey_darker,
)
from pyinspect.utils import dir_files

sys.path.append("./")

from experimental_validation._plot_utils import (
    draw_paws_steps,
    draw_mouse,
    mark_steps,
)
from experimental_validation._steps_utils import (
    print_steps_summary,
    stride_from_speed,
    get_steps,
)
from control.utils import from_json


"""
    Code to extracts steps data from tracking of mice running
"""

step_speed_th = 25  # cm / s
fps = 60
cm_per_px = 1 / 30.8

paws = ("left_fl", "right_fl", "left_hl", "right_hl")
paw_colors = {
    "left_fl": indigo_darker,
    "right_fl": salmon_darker,
    "left_hl": salmon,
    "right_hl": indigo,
}


def run():
    # Get paths
    data_folder = Path(
        "Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD"
    )
    save_folder = Path(
        "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\experimental_validation\\2WDD\\analysis"
    )

    trials_folder = data_folder / "TRIALS_CLIPS"
    tracking_folder = data_folder / "TRACKING_DATA"
    analysis_folder = data_folder / "ANALYSIS"
    analysis_folder.mkdir(exist_ok=True)

    trials_records = from_json(trials_folder / "trials_records.json")

    # Get files
    tracking_files = dir_files(trials_folder, "*_tracking.h5")
    logger.info(f"Starting steps analysis. Found {len(tracking_files)} files")
    # loop over files
    for n, path in enumerate(tracking_files):
        if n != 2:
            continue

        logger.info(f"Analyzing {path.name}")

        # Check that its a trial to be analyzed
        tname = path.stem.replace("_tracking", "")
        if (
            tname not in trials_records
            or trials_records[tname]["good"] == "tbd"
        ):
            logger.warning(
                f"Trial {tname}  not in trials records, dont know if should analyze. Skipping."
            )
            continue
        if not trials_records[tname]["good"]:
            logger.info(f"Trial {tname} is not to be analyzed")
            continue

        # get the tracking for the trial and for the whole session
        session_name = path.stem.split("_trial")[0] + "_video_tracking.h5"
        whole_session_tracking = pd.read_hdf(
            tracking_folder / session_name, key="hdf"
        )
        tracking = pd.read_hdf(path, key="hdf")

        # make figure
        (
            f,
            tracking_ax,
            paws_ax,
            cum_speeds_ax,
            ori_ax,
            turn_ax,
            turn_hist_ax,
            stride_hist_ax,
        ) = make_figure()

        # get when mouse crosses Y threshold
        y = t(tracking["body_y"], start=0, scale=False)
        try:
            start = np.where(y > 1400)[0][-1]
            end = np.where(y > 300)[0][-1]

            # check if the mouse stops in this interval
            bspeed = t(tracking["body_speed"], start=start, end=end) * fps
            if np.any(bspeed < 5):
                end = np.where(bspeed < 5)[0][0] + start

            logger.info(
                f"Start at frame: {start}/{len(y)} and end after {end - start} frames (frame {end})"
            )
        except IndexError:
            logger.info(
                "In this trial mouse didnt go below Y threshold, skipping"
            )
            continue

        # prep data
        LH_speed = t(tracking["left_hl_speed"], start=start, end=end) * fps
        LF_speed = t(tracking["left_fl_speed"], start=start, end=end) * fps
        RH_speed = t(tracking["right_hl_speed"], start=start, end=end) * fps
        RF_speed = t(tracking["right_fl_speed"], start=start, end=end) * fps
        body_speed = t(tracking["body_speed"], start=start, end=end) * fps
        orientation = t(
            r(tracking["body_whole_bone_orientation"]),
            scale=False,
            start=start,
            end=end,
        )

        speeds = dict(
            left_hl=LH_speed,
            left_fl=LF_speed,
            right_hl=RH_speed,
            right_fl=RF_speed,
        )

        # get steps
        (
            LH_steps,
            RF_steps,
            LF_steps,
            RH_steps,
            R_diagonal_steps,
            L_diagonal_steps,
            diagonal_steps,
            diag_data,
            step_starts,
        ) = get_steps(
            LH_speed, LF_speed, RH_speed, RF_speed, step_speed_th, start=start,
        )

        # draw mouse and steps
        draw_mouse(tracking_ax, tracking, whole_session_tracking, step_starts)
        draw_paws_steps(paw_colors, tracking_ax, tracking, step_starts, start)

        # Plot paw speeds
        for n, (paw, color) in enumerate(paw_colors.items()):
            if "_fl" in paw:
                alpha = 0.2
            else:
                alpha = 1

            paws_ax.plot(
                speeds[paw], color=color, lw=2, alpha=alpha, label=paw
            )
            cum_speeds_ax.plot(
                np.cumsum(speeds[paw]),
                color=color,
                lw=2,
                alpha=alpha,
                label=paw,
            )
        paws_ax.plot(
            body_speed,
            color=blue_grey_darker,
            lw=2,
            alpha=1,
            label="body",
            zorder=-1,
        )
        paws_ax.legend()
        cum_speeds_ax.legend()
        paws_ax.axhline(step_speed_th, lw=1, ls=":", color="k", zorder=-1)
        paws_ax.axhline(0, lw=2, color="k", zorder=1)

        # Plot orientation
        ori_ax.plot(
            orientation - orientation[0],
            label="body angle",
            color=blue_grey_darker,
            lw=4,
        )
        ori_ax.legend()

        # draw steps times
        for n, (paw, steps) in enumerate(
            zip(
                ("right_fl", "left_hl", "left_fl", "right_hl"),
                (RF_steps, LH_steps, LF_steps, RH_steps),
            )
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
        mark_steps(
            paws_ax,
            R_diagonal_steps.starts,
            R_diagonal_steps.ends,
            -80,
            "Diag.",
            5,
            noise=0,
            alpha=1,
            zorder=-1,
            lw=2,
            color="b",
        )

        mark_steps(
            paws_ax,
            L_diagonal_steps.starts,
            L_diagonal_steps.ends,
            -80 + -80 / 5,
            "Diag.",
            5,
            noise=0,
            alpha=1,
            zorder=-1,
            lw=2,
            color="r",
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

            if step["side"] == "L":
                left, right = step["paws"]
            else:
                right, left = step["paws"]

            r_stride = stride_from_speed(
                t(tracking[f"{right}_speed"]), step["start"], step["end"]
            )
            l_stride = stride_from_speed(
                t(tracking[f"{left}_speed"]), step["start"], step["end"]
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

        # plot stride length vs turn angle
        turn_ax.scatter(
            summary["stride_delta"],
            summary["angle_delta"],
            s=50,
            c=[1 if s == "R" else 0 for s in summary["side"]],
            zorder=10,
            lw=1,
            edgecolors=[0.2, 0.2, 0.2],
            cmap="bwr",
        )

        # plot histograms
        turn_hist_ax.hist(summary["angle_delta"], color=[0.3, 0.3, 0.3])
        turn_hist_ax.set(
            title=f"Total ammount turned: {np.sum(summary['angle_delta']):.1f}"
        )

        left_travel = np.mean([np.sum(LH_speed), np.sum(LF_speed)])
        right_travel = np.mean([np.sum(RH_speed), np.sum(RF_speed)])
        stride_hist_ax.hist(summary["stride_delta"], color=[0.3, 0.3, 0.3])
        stride_hist_ax.set(
            title=f"Total ammount travelled L:{left_travel:.1f} - R:{right_travel:.1f}"
        )

        # print steps data and save
        print_steps_summary(summary)
        pd.DataFrame(summary).to_hdf(save_folder / (tname + ".h5"), key="hdf")

        plt.show()
        break


# ----------------------------- utility functions ---------------------------- #


def make_figure():
    f = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = f.add_gridspec(3, 5)

    tracking_ax = f.add_subplot(gs[:, :2])
    tracking_ax.axis("equal")
    paws_ax = f.add_subplot(gs[0, 2:4])
    cum_speeds_ax = f.add_subplot(gs[1, 2:4])
    ori_ax = f.add_subplot(gs[2, 2:4])
    turn_ax = f.add_subplot(gs[0, 4])
    turn_hist_ax = f.add_subplot(gs[1, 4])
    stride_hist_ax = f.add_subplot(gs[2, 4])

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

    # tracking_ax.axis("off")
    paws_ax.set(title="paw speed", ylabel="speed\ncm/s", xlabel="Time\nframes")
    cum_speeds_ax.set(
        title="Dist.travelled", ylabel="Distance\ncm", xlabel="Time\nframes"
    )
    ori_ax.set(
        title="orientation", ylabel="ang\ndegrees", xlabel="Time\nframes"
    )
    turn_hist_ax.set(
        title="turn angle hist", ylabel="count", xlabel="turn angle"
    )
    stride_hist_ax.set(
        title="stride length hist", ylabel="count", xlabel="stride length"
    )

    # draw some lines on tracking ax
    # to denote arena areas
    for mark in (1850, 1600, 1300, 1050, 450, 380):
        tracking_ax.axhline(
            mark, lw=5, ls="--", color=[0.3, 0.3, 0.3], alpha=0.1, zorder=-1
        )

    set_figure_subplots_aspect(wspace=0.6, hspace=0.4)
    clean_axes(f)

    return (
        f,
        tracking_ax,
        paws_ax,
        cum_speeds_ax,
        ori_ax,
        turn_ax,
        turn_hist_ax,
        stride_hist_ax,
    )


def t(d, scale=True, start=0, end=-1):
    """
        Transform data by smoothing and
        going  from px to cm
    """
    try:
        d = d.values
    except Exception:
        pass

    d = rolling_mean(d[start:end], 5)

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


if __name__ == "__main__":
    run()
