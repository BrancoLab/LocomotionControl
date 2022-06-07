import numpy as np
import pandas as pd
from scipy.signal import medfilt
from fcutils.maths.signals import get_onset_offset
import sys
from loguru import logger
from dataclasses import dataclass
from myterial import light_blue_light, blue, indigo_dark, purple

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")

from data.dbase.db_tables import (
    Probe,
    Unit,
    Recording,
    Tracking,
    ProcessedLocomotionBouts2,
)

from data.data_utils import convolve_with_gaussian


# ---------------------------------------------------------------------------- #
#                                    CURVES                                    #
# ---------------------------------------------------------------------------- #
@dataclass
class Curve:
    s0: float  # position of "start"
    s: float  # position of apex
    sf: float  # position of end
    name: str
    color: str


curves = dict(
    first=Curve(0.0, 33.0, 40.0, "first", light_blue_light),
    second=Curve(45.0, 87, 96.0, "second", blue),
    third=Curve(98.0, 140.0, 144.0, "third", indigo_dark),
    fourth=Curve(150.0, 194.0, 205.0, "fourth", purple),
)


def get_roi_crossings(
    bouts: pd.DataFrame, curve: str, ds: int = 30, direction="out"
) -> pd.DataFrame:
    """
        Get all the times the mouse crossed a curve ROI in the outward/inwrard direction, 
        from bouts
    """
    crossings = dict(
        enter_frame=[],
        exit_frame=[],
        bout_start_frame=[],
        bout_end_frame=[],
        session_start_frame=[],
        session_end_frame=[],
        at_apex=[],
        bout_idx=[],
    )

    # get all the times the mouse goes through a curve
    sapex = curves[curve].s

    # for each bout, get the frame at which the mouse enters and exit
    for i, bout in bouts.iterrows():
        S = np.array(bout.s)
        if direction == "out":
            enter = np.where(S < (sapex - ds))[0]
        else:
            enter = np.where(S > (sapex + ds))[0]

        if not len(enter):
            enter = 0
        else:
            enter = enter[-1]

        if direction == "out":
            exit = np.where(S[enter:] < sapex + ds)[0] + enter
        else:
            exit = np.where(S[enter:] > sapex - ds)[0] + enter

        if not len(exit):
            continue
        else:
            exit = exit[-1]

        if exit < enter:
            raise ValueError(f"exit before enter: {exit} < {enter}")

        if abs(exit - enter) > 100:
            continue

        # if np.any(np.array(bout.speed[enter:exit]) < 5):
        #     continue

        # check that the entiire ROI is covered in the bout
        if direction == "out":
            if bout.s[enter] > sapex - ds + 5:
                continue
            if bout.s[exit] < sapex + ds - 5:
                continue
        else:
            if bout.s[enter] < sapex + ds - 5:
                continue
            if bout.s[exit] > sapex - ds + 5:
                continue

        # get when at curve apex
        d = (S[enter:exit] - sapex) ** 2
        idx = np.argmin(d)

        crossings["enter_frame"].append(enter)
        crossings["exit_frame"].append(exit)
        crossings["bout_start_frame"].append(bout.start_frame)
        crossings["bout_end_frame"].append(bout.end_frame)
        crossings["session_start_frame"].append(enter + bout.start_frame)
        crossings["session_end_frame"].append(exit + bout.start_frame)
        crossings["at_apex"].append(enter + idx + bout.start_frame)
        crossings["bout_idx"].append(i)
    return pd.DataFrame(crossings)


# ---------------------------------------------------------------------------- #
#                                   GET DATA                                   #
# ---------------------------------------------------------------------------- #


def get_recording_names():
    """
        Get the names of the recordings (M2)
    """
    return (Recording * Probe & "target='MOs'").fetch("name")


def get_speed(x, y):
    """
        Compute speed at each frame from XY coordinates, somehow 
        its missing for paws in database
    """
    rawspeed = (
        np.hstack([[0], np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)]) * 60
    )
    return convolve_with_gaussian(rawspeed, 9)


def get_data(recording: str):
    """
        Get all relevant data for a recording.
        Gets the ephys data and tracking data for all limbs
    """
    tracking = Tracking.get_session_tracking(
        recording, body_only=False, movement=True
    )

    if len(tracking) == 0:
        logger.warning(f"No tracking data for {recording}")
        return None, None, None, None, None, None

    left_fl = tracking.loc[tracking.bpname == "left_fl"].iloc[0]
    right_fl = tracking.loc[tracking.bpname == "right_fl"].iloc[0]
    left_hl = tracking.loc[tracking.bpname == "left_hl"].iloc[0]
    right_hl = tracking.loc[tracking.bpname == "right_hl"].iloc[0]
    body = tracking.loc[tracking.bpname == "body"].iloc[0]

    logger.info(f"Got tracking data for {recording}")

    # get units
    recording = (Recording & f"name='{recording}'").fetch1()
    cf = recording["recording_probe_configuration"]
    units = Unit.get_session_units(
        recording["name"], cf, spikes=True, firing_rate=True, frate_window=100,
    )
    if len(units):
        units = units.sort_values("brain_region", inplace=False).reset_index()
        logger.info(f"Got {len(units)} units for {recording['name']}")
    else:
        logger.info(f"No units for {recording['name']}")

    return units, left_fl, right_fl, left_hl, right_hl, body


def get_session_bouts(
    session: str, complete: str = "true", direction: str = "outbound"
):
    """
        Get bouts (complete/all - in/out bound) for a session
    """
    query = ProcessedLocomotionBouts2 & f'name="{session}"'
    if complete is not None:
        query = query & f"complete='{complete}'"

    if direction is not None:
        query = query & f"direction='{direction}'"

    bouts = pd.DataFrame(query.fetch())
    logger.info(
        f"Got {len(bouts)} bouts for {session} | {complete} | {direction}"
    )
    return bouts


def cleanup_running_bouts(bouts, tracking, min_delta_gcoord=0.5):
    """
        Remove bouts that are too short
        and cleans up start/end times
    """
    # filter bouts
    bouts = bouts.loc[
        (bouts.gcoord_delta > min_delta_gcoord)
        & (bouts.direction == "outbound")
        & (bouts.duration < 15)
    ]

    correct_start_frame, correct_stop_frame = [], []
    keep = []
    for i, bout in bouts.iterrows():
        gcoord = tracking.global_coord[bout.start_frame : bout.end_frame] * 260
        gcoord = medfilt(gcoord, 11)

        gf = np.max(gcoord)
        stop = np.where(gcoord >= gf - 1)[0][0]
        start = np.argmin(gcoord[:stop])

        if np.max(np.diff(gcoord[start:stop])) < 10:
            keep.append(i)

            correct_start_frame.append(start)
            correct_stop_frame.append(stop)

    bouts = bouts.loc[keep]

    bouts["start_frame"] = np.array(correct_start_frame) + bouts.start_frame
    bouts["stop_frame"] = np.array(correct_stop_frame) + bouts.start_frame

    # sort but gcoord at start frame
    bouts["gcoord0"] = [tracking.global_coord[f] for f in bouts.start_frame]
    bouts = bouts.sort_values("gcoord0").reset_index()
    return bouts


# ---------------------------------------------------------------------------- #
#                                 WALKING ONSET                                #
# ---------------------------------------------------------------------------- #
def get_walking_from_paws(left_fl, right_fl, left_hl, right_hl, SPEED_TH):
    """
        "walking" is when the paws are moving
    """

    limbs_moving = {
        l.bpname: medfilt(l.speed, 31) > SPEED_TH
        for l in (left_fl, right_fl, left_hl, right_hl)
    }

    walking = np.sum(np.vstack(list(limbs_moving.values())), axis=0)
    walking = walking >= 2  # at least 2 limbs moving
    walking = medfilt(walking.astype(float), 15)
    return walking


def get_walking_from_body(body, SPEED_TH):
    """
        "walking" is when the body is moving
    """
    walking = medfilt(body.speed, 11) > SPEED_TH
    walking = medfilt(walking.astype(float), 11)
    return walking


def get_clean_walking_onsets(walking, MIN_WAKING_DURATION, MIN_PAUSE_DURATION):
    """
        Keep only locomotion onset when it lasts long enough and there's a pause before the previous one
    """
    onsets, offsets = get_onset_offset(walking, 0.5)
    print(f"Foun {len(onsets)} raw walking bouts ({len(offsets)} offsets)")

    walking_starts, walking_ends = [], []
    for onset in onsets:
        # get the last offset before this
        prev_offset = offsets[offsets < onset]
        if len(prev_offset) == 0:
            continue
        else:
            prev_offset = prev_offset[-1]

        # get the next offset after this
        next_offset = offsets[offsets > onset]
        if len(next_offset) == 0:
            continue
        else:
            next_offset = next_offset[0]

        # get pause and bout duration
        pause_duration = (onset - prev_offset) / 60
        bout_duration = (next_offset - onset) / 60

        # check conditions
        if (
            pause_duration < MIN_PAUSE_DURATION
            or bout_duration < MIN_WAKING_DURATION
        ):
            # print(f"Skipping bout at {onset}, duration {bout_duration} pause {pause_duration}")
            continue

        # keep onsets offests
        walking_starts.append(onset)
        walking_ends.append(next_offset)

    print(f"Kept {len(walking_starts)} valid walking bouts")
    return np.array(walking_starts) / 60, np.array(walking_ends) / 60


# ---------------------------------------------------------------------------- #
#                                 TUNING CURVES                                #
# ---------------------------------------------------------------------------- #
def bin_variable(x, bins=10):
    """
    Bin variable x into bins
    and return which frames are in which bin and the bin values
    """

    x = medfilt(x, kernel_size=11)

    # keep only frames within 95th CI
    # low, high = np.percentile(x, [0.5, 99.5])
    # x = x[(x >= low) & (x <= high)]
    if isinstance(bins, int):
        n_bins = bins
    else:
        n_bins = len(bins) - 1

    _, edges = np.histogram(x, bins=bins)

    in_bin = dict()
    bin_values = []
    for i in range(n_bins):
        in_bin[i] = np.where((x > edges[i]) & (x <= edges[i + 1]))[0]
        bin_values.append(edges[i] + 0.5 * (edges[i + 1] - edges[i]))

    return in_bin, bin_values


def get_frate_per_bin(unit, in_bin) -> dict:
    """
        Get firing rate per bin by taking all the spikes in a bin and dividing by the number of frames in that bin
    """
    n_ms_per_frame = 1000 / 60

    # get which unit spikes are in which bin
    in_bin_frate = {}
    for i, bin_frames in in_bin.items():
        n_spikes = len(unit.spikes[np.isin(unit.spikes, bin_frames)])
        n_seconds = len(bin_frames) * n_ms_per_frame / 1000
        in_bin_frate[i] = n_spikes / n_seconds if n_seconds > 0 else np.nan
    return in_bin_frate
