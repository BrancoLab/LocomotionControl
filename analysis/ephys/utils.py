import numpy as np
import pandas as pd
from scipy.signal import medfilt

from fcutils.maths.signals import get_onset_offset


from data.data_utils import convolve_with_gaussian
from data.dbase.db_tables import Probe, Unit, Tracking


# ---------------------------------------------------------------------------- #
#                                   GET DATA                                   #
# ---------------------------------------------------------------------------- #
def get_speed(x, y):
    """
        Compute speed at each frame from XY coordinates, somehow 
        its missing for paws in database
    """
    rawspeed = (
        np.hstack([[0], np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)]) * 60
    )
    return convolve_with_gaussian(rawspeed, 9)


def get_data(REC):
    tracking = Tracking.get_session_tracking(REC, body_only=False)

    units = pd.DataFrame(
        Unit * Unit.Spikes * Probe.RecordingSite & f'name="{REC}"'
    )
    units = units.sort_values("brain_region", inplace=False).reset_index()

    left_fl = tracking.loc[tracking.bpname == "left_fl"].iloc[0]
    right_fl = tracking.loc[tracking.bpname == "right_fl"].iloc[0]
    left_hl = tracking.loc[tracking.bpname == "left_hl"].iloc[0]
    right_hl = tracking.loc[tracking.bpname == "right_hl"].iloc[0]
    body = tracking.loc[tracking.bpname == "body"].iloc[0]

    for limb in (left_fl, right_fl, left_hl, right_hl):
        limb.speed = get_speed(limb.x, limb.y)

    return units, left_fl, right_fl, left_hl, right_hl, body


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
