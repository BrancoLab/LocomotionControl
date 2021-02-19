from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import sys
import numpy as np

from fcutils.plot.figure import (
    set_figure_subplots_aspect,
    clean_axes,
)
from fcutils.plot.figure import save_figure
from myterial import blue_grey_darker
from fcutils.progress import track
from fcutils.maths.signals import rolling_mean
from fcutils.path import files

sys.path.append("./")

from experimental_validation._plot_utils import draw_tracking
from experimental_validation.trials import Trials
from experimental_validation._tracking import cm_per_px

from kinematics.steps import Steps
from kinematics.plot_utils import draw_mouse, mark_steps, draw_paws_steps_XY
from kinematics.fixtures import PAWS_COLORS
from kinematics._steps import print_steps_summary, stride_length

"""
    Code to extracts steps data from tracking of mice running
"""


fps = 60
step_speed_th = 25  # cm / s | steps are considered if peak speed > this
precise_th = 9  # cm / s | to find precise step times


def extract_all(save_folder):
    trials = Trials()

    # loop over trials
    logger.info(f"Starting steps analysis of {trials}")
    for n, trial in track(enumerate(trials), total=len(trials)):
        logger.info(f"Analyzing {trial}")
        if not trial.has_tracking or not trial.good:
            logger.info("Trial doesnt have tracking or is bad.")
            continue

        # smooth things
        trial.orientation = rolling_mean(trial.orientation, 4)
        trial.v = rolling_mean(trial.v, 4)
        trial.speeds = {k: rolling_mean(v, 4) for k, v in trial.speeds.items()}

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

        # get steps
        steps = Steps(trial)
        try:
            (
                swing_phases,
                R_diagonal_steps,
                L_diagonal_steps,
                diagonal_steps,
                diag_data,
                step_starts_R,
                step_starts_L,
            ) = steps.extract_steps(step_speed_th, precise_th=precise_th)
        except Exception:
            logger.warning(f"Failed to extract tep info from trial {trial}")
            continue

        # draw mouse and steps
        if trial.whole_session_tracking is not None:
            draw_tracking(trial.whole_session_tracking, tracking_ax)

        draw_mouse(
            trial, tracking_ax, step_starts_R,
        )

        draw_paws_steps_XY(trial, tracking_ax, step_starts_R)

        # Plot paw speeds
        for n, (paw, color) in enumerate(PAWS_COLORS.items()):
            if "_fl" in paw:
                alpha = 0.2
            else:
                alpha = 1

            paws_ax.plot(
                trial.speeds[paw], color=color, lw=2, alpha=alpha, label=paw
            )
            cum_speeds_ax.plot(
                np.cumsum(trial.speeds[paw]),
                color=color,
                lw=2,
                alpha=alpha,
                label=paw,
            )
        paws_ax.plot(
            trial.v,
            color=blue_grey_darker,
            lw=2,
            alpha=1,
            label="body",
            zorder=-1,
        )
        cum_speeds_ax.legend()
        paws_ax.axhline(step_speed_th, lw=2, ls="--", color="k", zorder=-1)
        paws_ax.axhline(precise_th, lw=1, ls=":", color="k", zorder=-1)

        paws_ax.axhline(0, lw=2, color="k", zorder=1)

        # Plot orientation delta
        ori_ax.plot(
            trial.orientation - trial.orientation[0],
            label="body angle delta",
            color=blue_grey_darker,
            lw=4,
        )
        ori_ax.legend()

        # draw steps times
        for n, (paw, steps) in enumerate(swing_phases.items()):
            if "hl" in paw:
                alpha = 1
            else:
                alpha = 0.2
            mark_steps(
                paws_ax,
                steps.starts,
                steps.ends,
                -10 * (n + 1),
                5,
                color=PAWS_COLORS[paw],
                alpha=alpha,
                zorder=-1,
                lw=2,
            )

        # draw diagonal steps
        mark_steps(
            paws_ax,
            R_diagonal_steps.starts,
            R_diagonal_steps.ends,
            -80,
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
        )
        for n, step in diag_data.items():
            # stride delta

            if step["side"] == "L":
                left, right = step["paws"]
            else:
                right, left = step["paws"]

            r_stride = stride_length(
                trial.right_hl, step["start"], step["end"]
            )
            l_stride = stride_length(trial.left_hl, step["start"], step["end"])

            summary["stride_delta"].append(l_stride - r_stride)

            # angle delta
            turn = (
                trial.orientation[step["end"]]
                - trial.orientation[step["start"]]
            )
            summary["angle_delta"].append(turn)

            # more info
            summary["number"].append(n)
            summary["side"].append(step["side"])
            summary["start"].append(step["start"])
            summary["end"].append(step["end"])

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

        left_travel = np.mean(
            [np.sum(trial.left_hl.speed), np.sum(trial.left_fl.speed)]
        )
        right_travel = np.mean(
            [np.sum(trial.right_hl.speed), np.sum(trial.right_fl.speed)]
        )
        stride_hist_ax.hist(summary["stride_delta"], color=[0.3, 0.3, 0.3])
        stride_hist_ax.set(
            title=f"Total ammount travelled L:{left_travel:.1f} - R:{right_travel:.1f}"
        )

        # print steps data and save
        print_steps_summary(summary)
        pd.DataFrame(summary).to_hdf(
            save_folder / (trial.name + "_extracted_steps.h5"), key="hdf"
        )
        save_figure(f, save_folder / (trial.name + "_extracted_steps"))
        plt.close(f)
        del f

        # plt.show()
        # break


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

    x, y = 10, 45  # axes lims for summary plot
    turn_ax.set(  # summary ax
        title="stride difference vs turn angle",
        # xlim=[-x, x],
        xticks=[-x, 0, x],
        xticklabels=[f"R>L\n{-x}", "R=L\n0", f"R<L\n{x}"],
        xlabel="(L-R) stride-delta\m(cm)",
        # ylim=[-y, y],
        yticks=[-y, 0, y],
        yticklabels=[f"turn\nleft\n{-y}", "no\nturn\n0", f"turn\nright\n{y}"],
        ylabel="(end-start) angle-delta\n(deg)",
    )
    turn_ax.axvline(0, ls="--", color=[0.5, 0.5, 0.5])
    turn_ax.axhline(0, ls="--", color=[0.5, 0.5, 0.5])

    tracking_ax.set(xlim=[-10, 45], ylim=[0, 65], xlabel="cm", ylabel="cm")

    # tracking_ax.axis("off")
    paws_ax.set(title="paw speed", ylabel="speed\ncm/s", xlabel="Time\nframes")
    cum_speeds_ax.set(
        title="Dist.travelled", ylabel="Distance\ncm", xlabel="Time\nframes"
    )
    ori_ax.set(
        title="orientation", ylabel="$\Delta angle$", xlabel="Time\nframes"
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
            mark * cm_per_px,
            lw=5,
            ls="--",
            color=[0.3, 0.3, 0.3],
            alpha=0.1,
            zorder=-1,
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


def steps_summary(trials_folder):
    """
        Loads steps data for each individual trial and makes a summary plot
    """
    data = []
    for fpath in files(trials_folder, "*steps.h5"):
        data.append(pd.read_hdf(fpath, key="hdf"))

    # steps = pd.concat(data).reset_index()
    f, ax = plt.subplots(figsize=(10, 10))


if __name__ == "__main__":
    save_folder = Path(
        r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\experimental_validation\2WDD\analysis\trials_steps"
    )
    # extract_all(save_folder)

    steps_summary(save_folder)
