from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import sys
import numpy as np
from rich import print

from fcutils.plotting.utils import (
    set_figure_subplots_aspect,
    clean_axes,
    save_figure,
)
from fcutils.maths.utils import rolling_mean
from myterial import (
    salmon_darker,
    indigo_darker,
    indigo,
    salmon,
    blue_grey_darker,
    orange,
)
from pyinspect.utils import dir_files

sys.path.append('./')

from experimental_validation._plot_utils import draw_paws_steps, draw_mouse, mark_steps
from experimental_validation._steps_utils import (
    get_steps,
    print_steps_summary,
    stride_from_speed,
)


'''
    Code to extracts steps data from tracking of mice running
'''

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
    folder = Path('Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD')
    trials_folder = folder / 'TRIALS_CLIPS'
    tracking_folder = folder / 'TRACKING_DATA'
    analysis_folder = folder / 'ANALYSIS'
    analysis_folder.mkdir(exist_ok=True)

    # Get files
    tracking_files = dir_files(trials_folder, '*_tracking.h5')
    logger.info(f'Starting steps analysis. Found {len(tracking_files)} files') 
    # loop over files
    for path in tracking_files:
        logger.info(f'Analyzing {path.name}')

        # get the tracking for the trial and for the whole session
        session_name = path.stem.split('_trial')[0] + '_video_tracking.h5'
        whole_session_tracking = pd.read_hdf(tracking_folder / session_name, key='hdf')
        tracking = pd.read_hdf(path, key='hdf')

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
        y = t(tracking["body_y"], start=0)
        try:
            start = np.where(y <= 1300)[0][-]
            logger.info(f'Start at frame: {start}')
        except  ValueError:
            logger.info('In this trial mouse didnt go below Y threshold, skipping')
            continue

        # prep data
        LH_speed = t(tracking["left_hl_speed"], start=start) * fps
        LF_speed = t(tracking["left_fl_speed"], start=start) * fps
        RH_speed = t(tracking["right_hl_speed"], start=start) * fps
        RF_speed = t(tracking["right_fl_speed"], start=start) * fps


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
            LH_speed, LF_speed, RH_speed, RF_speed, step_speed_th, start=start
        )

        # draw mouse and steps
        draw_mouse(tracking_ax, tracking, whole_session_tracking, step_starts)
        draw_paws_steps(paw_colors, tracking_ax, tracking, step_starts, start)

        # # Plot paw speeds
        # for n, (paw, color) in enumerate(paw_colors.items()):
        #     if paw in ("LH", "RH"):
        #         alpha = 0.2
        #     else:
        #         alpha = 1
        #     spd = t(tracking[f"{paw}_speed"]) * fps

        #     paws_ax.plot(spd, color=color, lw=2, alpha=alpha, label=paw)
        #     cum_speeds_ax.plot(
        #         np.cumsum(spd), color=color, lw=2, alpha=alpha, label=paw
        #     )
        # paws_ax.legend()
        # cum_speeds_ax.legend()
        # paws_ax.axhline(step_speed_th, lw=1, ls=":", color="k", zorder=-1)
        # paws_ax.axhline(0, lw=2, color="k", zorder=1)

        # # Plot orientation
        # orientation = t(r(tracking["body_whole_bone_orientation"]), scale=False)
        # ori_ax.plot(
        #     orientation - orientation[0],
        #     label="body angle",
        #     color=blue_grey_darker,
        #     lw=4,
        # )
        # ori_ax.legend()

        # # draw steps times
        # for n, (paw, steps) in enumerate(
        #     zip(("right_fl", "left_hl", "left_fl", "right_hl"), (RF_steps, LH_steps, LF_steps, RH_steps))
        # ):
        #     mark_steps(
        #         paws_ax,
        #         steps.starts,
        #         steps.ends,
        #         -10 * (n + 1),
        #         paw,
        #         5,
        #         color=paw_colors[paw],
        #         alpha=0.9,
        #         zorder=-1,
        #         lw=2,
        #     )

        # # draw diagonal steps
        # mark_steps(
        #     paws_ax,
        #     R_diagonal_steps.starts,
        #     R_diagonal_steps.ends,
        #     -80,
        #     "Diag.",
        #     5,
        #     noise=0,
        #     alpha=1,
        #     zorder=-1,
        #     lw=2,
        #     color="b",
        # )

        # mark_steps(
        #     paws_ax,
        #     L_diagonal_steps.starts,
        #     L_diagonal_steps.ends,
        #     -80 + -80 / 5,
        #     "Diag.",
        #     5,
        #     noise=0,
        #     alpha=1,
        #     zorder=-1,
        #     lw=2,
        #     color="r",
        # )


        plt.show()
        break



# ----------------------------- utility functions ---------------------------- #


def make_figure():
    f = plt.figure(constrained_layout=True, figsize=(16, 9))
    gs = f.add_gridspec(3, 6)

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
        tracking_ax.axhline(mark, lw=5, ls='--', color=[0.3, 0.3, 0.3], alpha=.1, zorder=-1)


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


def t(d, scale=True, start=0):
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

if __name__ == '__main__':
    run()