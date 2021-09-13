from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

from fcutils.maths.signals import get_onset_offset

from data.data_structures import TrackingData
from data import colors


def plot_speeds(
    body_speed,
    start: int = 0,
    end: int = -1,
    is_moving: np.ndarray = None,
    ax: plt.axis = None,
    show: bool = True,
    **other_speeds,
):
    """
        Just plot some speeds for debugging stuff
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(14, 8))

    ax.plot(body_speed[start:end], label="body", color="salmon")
    for name, speed in other_speeds.items():
        ax.plot(speed[start:end], label=name)

    if is_moving is not None:
        ax.plot(is_moving, lw=4, color="k")

    ax.legend()
    ax.set(xlabel="time (frames)", ylabel="speed (cm/s)")
    if show:
        plt.show()


def plot_bout_start_end(
    body_speed,
    start: int = 0,
    end: int = -1,
    is_moving: np.ndarray = None,
    precise_end: int = None,
    precise_start: int = None,
    speed_th: float = 1,
    **other_speeds,
):
    """
        Plot speeds zoomed in at start and end of bout
    """
    f, axes = plt.subplots(ncols=2, sharey=True, figsize=(16, 9))

    # plot start & end
    plot_speeds(
        body_speed,
        start=start - 60 if start > 60 else 0,
        end=start + 60,
        ax=axes[0],
        show=False,
        **other_speeds,
    )
    plot_speeds(
        body_speed,
        start=end - 60,
        end=end + 60,
        ax=axes[1],
        show=False,
        **other_speeds,
    )

    # mark precise starts and ends
    if precise_start is not None:
        axes[0].axvline(
            60 + (precise_start - start), lw=4, alpha=0.5, color="salmon"
        )
    if precise_end is not None:
        axes[1].axvline(
            60 + (precise_end - end), lw=4, alpha=0.5, color="salmon"
        )

    for name, ax in zip(("start", "end"), axes):
        ax.axvline(60, ls="--", color=[0.4, 0.4, 0.4], zorder=-1)
        ax.axhline(speed_th, color=[0.4, 0.4, 0.4], zorder=-1)
        ax.set(title=name)
    plt.show()


def get_when_moving(
    body_speed: np.ndarray,
    speed_th: float,
    max_pause: float,
    min_duration: float,
    min_peak_speed: float,
) -> np.ndarray:
    """
        Get when the mouse is moving by looking at when the speed is above a given threshold

        Arguments:
            body_speed: speed at each frame
            speed_th: when speed > th the mouse is moving
            max_pause: if speed < th for less than max_pause ignore that as a pause.
            min_duration: keep only bouts that last at least this long
            min_peak_speed: each bout must reach at least this speed
    """
    # get when the mouse is moving
    moving = np.where(body_speed > speed_th)[0]
    is_moving = np.zeros_like(body_speed)
    is_moving[moving] = 1

    # remove pauses
    onsets, offsets = get_onset_offset(is_moving, 0.5)
    for n, onset in enumerate(onsets):
        if n > 0:
            # get the duration of the pause from the last offset
            offset = offsets[n - 1]
            if onset - offset < max_pause * 60:
                # pause was too short
                is_moving[offset - 1 : onset + 1] = 1

    # remove bouts that are too short
    onsets, offsets = get_onset_offset(is_moving, 0.5)
    for onset, offset in zip(onsets, offsets):
        reached_speed = (
            len(np.where(body_speed[onset:offset] >= min_peak_speed)[0]) > 0
        )
        if offset - onset < min_duration * 60 or not reached_speed:
            # bout was too short or too slow
            is_moving[onset - 1 : offset + 1] = 0

    return is_moving


def get_threshold_crossing(
    tracking: tuple, frame: int, direction: str, threshold: float
) -> int:
    """
        Gets the last/first time the speed of all paws crossed a threshold
    """
    speeds = (
        tracking.right_fl.speed,
        tracking.left_fl.speed,
        tracking.right_hl.speed,
        tracking.left_hl.speed,
    )
    if direction == "up":
        if frame == 0:
            return frame
        start = frame - 60 if frame > 60 else 0

        crosses = []
        for paw in speeds:
            try:
                crosses.append(np.where(paw[start:frame] < threshold)[0][-1])
            except IndexError:
                crosses.append(np.nan)

        val = np.nanmin(crosses) + start
        if val > frame or np.isnan(val):
            logger.warning("Error in precise threshold crossing computation")
            return None
        return int(val)
    else:
        n_frames = len(tracking.right_fl.speed)
        if frame == n_frames:
            return frame
        end = frame + 60 if frame < (n_frames - 60) else n_frames

        crosses = []
        for paw in speeds:
            try:
                crosses.append(np.where(paw[frame:end] < threshold)[0][0])
            except IndexError:
                crosses.append(np.nan)

        val = np.nanmax(crosses) + frame
        if val > end or np.isnan(val):
            logger.warning("Error in precise threshold crossing computation")
            return None
        return int(val)


def get_bout_start_end_times(
    tracking: tuple, bstart: int, bend: int
) -> Tuple[int, int]:
    """
        Look at the movement of the paws to decide exactly when the bout starts
    """

    # get exactly when the bout starts and ends
    precise_start = get_threshold_crossing(tracking, bstart, "up", 2)
    precise_end = get_threshold_crossing(tracking, bend, "down", 2)

    # plot_bout_start_end(tracking.body.speed,
    #     rfl=tracking.right_fl.speed,
    #     lfl=tracking.left_fl.speed,
    #     rhl=tracking.right_hl.speed,
    #     lhl=tracking.left_hl.speed,
    #     precise_start = precise_start,
    #     precise_end = precise_end,
    #     start=bstart, end=bend, speed_th=speed_th)

    return precise_start, precise_end


def check_gcoord_delta(
    tracking, bstart: int, bend: int, min_gcoord_delta: float
) -> bool:
    gstart, gend = (
        tracking.body.global_coord[bstart],
        tracking.body.global_coord[bend],
    )
    return abs(gend - gstart) >= min_gcoord_delta


def get_bout_direction(tracking: tuple, bstart: int, bend: int) -> str:
    # plt.plot(tracking.body.global_coord[bstart:bend]); plt.show()
    gstart, gend = (
        tracking.body.global_coord[bstart],
        tracking.body.global_coord[bend],
    )
    if gend > gstart:
        return "outbound"
    else:
        return "inbound"


def get_bout_complete_and_rois(
    tracking: tuple, bstart: int, bend: int
) -> Tuple[str, int, int]:
    # check if it was a complete trip
    gstart, gend = (
        tracking.body.global_coord[bstart],
        tracking.body.global_coord[bend],
    )
    complete = abs(gend - gstart) > 0.8

    # get start and end rois
    segstart = tracking.body.segment[bstart]
    segend = tracking.body.segment[bend]

    return "true" if complete else "false", segstart, segend


def get_session_bouts(
    key: dict,
    tracking: pd.DataFrame,
    is_hairpin: bool,
    speed_th: float,
    max_pause: float,
    min_duration: float,
    min_peak_speed: float,
    min_gcoord_delta: float,
) -> list:
    """
        Gets all the locomotion bouts for an experimental session
    """
    tracking = TrackingData.from_dataframe(tracking)

    # get when the mouse is moving
    is_moving = get_when_moving(
        tracking.body.speed, speed_th, max_pause, min_duration, min_peak_speed
    )
    # plot_speeds(tracking.body.speed, is_moving=is_moving)

    # get bouts onsets and offsets
    onsets, offsets = get_onset_offset(is_moving, 0.5)
    logger.debug(f"Found {len(onsets)} bouts")

    # fill up all details
    bouts = []
    for bstart, bend in zip(onsets, offsets):
        if bend < bstart:
            raise ValueError("Something went wrong...")

        # get precise bout start and end times based on paw speeds
        bstart, bend = get_bout_start_end_times(tracking, bstart, bend)
        if bstart is None or bend is None:
            continue
        elif bend == len(tracking.body.x):
            bend -= 1

        # remove bouts too close to start and end of recording
        if bstart < 60 or bend > (len(tracking.body.speed) - 60):
            continue

        if is_hairpin:
            # check there's enough change in global coordinate
            if not check_gcoord_delta(
                tracking, bstart, bend, min_gcoord_delta
            ):
                continue

            # get bout direction of movement if hairpin
            direction = get_bout_direction(tracking, bstart, bend)

            # get complete and ROIs
            complete, start_roi, end_roi = get_bout_complete_and_rois(
                tracking, bstart, bend
            )
        else:
            direction, complete = "none", "none"
            start_roi, end_roi = -1, -1

        if bstart in [b["start_frame"] for b in bouts]:
            continue

        # put everything together
        bout = key.copy()
        bout["start_frame"] = bstart
        bout["end_frame"] = bend
        bout["duration"] = (bend - bstart) / 60
        bout["direction"] = direction
        bout["color"] = colors.bout_direction_colors[direction]
        bout["complete"] = complete
        bout["start_roi"] = start_roi
        bout["end_roi"] = end_roi
        bouts.append(bout)

    n_complete = len([b for b in bouts if b["complete"] == "true"])
    logger.debug(
        f" kept {len(bouts)}/{len(onsets)} ({len(bouts)/len(onsets):.2f} %) bouts of which {n_complete} are complete"
    )
    return bouts
