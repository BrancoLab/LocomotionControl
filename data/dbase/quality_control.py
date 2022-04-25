import numpy as np
from pathlib import Path
from loguru import logger
import sys

sys.path.append("./")

import matplotlib.pyplot as plt


from fcutils.video import get_video_params
from fcutils.maths.signals import get_onset_offset

from data.dbase.io import load_bin, get_recording_local_copy
from data.debug_utils import plot_recording_triggers


def load_or_open(
    base_path: str, data_type: str, bin_file: Path, idx: int, **kwargs
):
    """
        Tries to load a previously saved .npy file with some relevant data,
        otherwise it opens a .bin file and saves it to .npy for future use
    """
    savepath = Path(base_path).parent / (
        Path(base_path).stem + f"_{data_type}_sync.npy"
    )
    if savepath.exists():
        logger.debug("Loading previously extracted signal from .npy")
        return np.load(savepath)
    else:
        logger.debug("Opening binary and saving to numpy")
        binary = load_bin(str(bin_file), **kwargs)
        signal = binary[:, idx].copy()
        np.save(savepath, signal)
        return signal


def validate_behavior(video_file_path, ai_file_path, analog_sampling_rate):
    """
        Checks that a bonsai behaviour recording ran correctly
        (i.e. no frames dropped)

        Arguments:
            video_file_pat: str. Path to .avi video file
            ai_file_pat: str. Path to .bin analog inputs file
            analog_sampling_rate: int. Sampling rate in bonsai
    """

    def _get_triggers(nsigs=4):
        # load analog
        analog = load_bin(ai_file_path, nsigs=nsigs)
        logger.debug(
            f"Getting triggers with: {nsigs} signals in .bin file | analog shape {analog.shape}"
        )

        # check that the number of frames is correct
        frame_trigger_times = get_onset_offset(analog[:, 0], 4.5)[0]
        return frame_trigger_times

    name = Path(video_file_path).name
    logger.debug(f"Running validate BEHAVIOR on {name}")

    # load video and get metadata
    try:
        nframes, w, h, fps, _ = get_video_params(video_file_path)
    except ValueError:
        logger.warning("Could not open video file")
        return False, 0, 0, 0, 0, 0, "couldnt_open_video"

    if fps != 60:
        raise ValueError("Expected video FPS: 60")

    try:
        frame_trigger_times = _get_triggers()
    except ValueError as e:
        logger.warning(
            f"While validating bonsai for {name} could not open binary file {ai_file_path}:\n''{e}''"
        )
        return False, 0, 0, 0, 0, 0, "no_bonsai_file_to_open"

    if len(frame_trigger_times) - nframes:
        logger.debug(
            f"Extracting frame stims with 4 signals in .bin file found the wrong number of triggers ({len(frame_trigger_times)} triggers instead of {nframes} frames)"
        )
        if abs(nframes - len(frame_trigger_times)) > 100:
            try:
                nsigs = 3
                frame_trigger_times = _get_triggers(nsigs=nsigs)
                if len(frame_trigger_times) != nframes:
                    raise ValueError
            except ValueError:
                logger.warning(
                    f"session: {name} - found {nframes} video frames and {len(frame_trigger_times)} trigger times in analog input"
                )
                return (
                    False,
                    nsigs,
                    0,
                    0,
                    0,
                    0,
                    f"wrong_number_of_triggers_and_frames_{len(frame_trigger_times)}vs{nframes}",
                )
        else:
            logger.warning(
                "Something went very wrong, couldnt figure out the right number of triggers"
            )
            return False, 4, 0, 0, 0, 0, "wrong_number_of_triggers_and_frames"
    else:
        nsigs = 4
        logger.debug(
            f"{name} has {nframes} frames and {len(frame_trigger_times)} trigger times were found"
        )

    # check that the number of frames is what you'd expect given the duration of the exp
    first_frame_s = frame_trigger_times[0] / analog_sampling_rate
    last_frame_s = frame_trigger_times[-1] / analog_sampling_rate

    exp_dur = last_frame_s - first_frame_s  # video duration in seconds
    expected_n_frames = np.round(exp_dur * 60).astype(np.int64)
    if np.abs(expected_n_frames - nframes) > 5:
        raise ValueError(
            f"[b yellow]Expected {expected_n_frames} frames but found {nframes} in video"
        )
    logger.debug(f"{name} video duration is correct")

    return (
        True,
        nsigs,
        exp_dur,
        nframes,
        frame_trigger_times[0],
        frame_trigger_times[1],
        "behav_valid",
    )


# ---------------------------------------------------------------------------- #
#                              validate ephys data                             #
# ---------------------------------------------------------------------------- #


def get_onsets_offsets(bonsai_probe_sync, ephys_probe_sync, sampling_rate):
    logger.debug("extracting sync signal pulses")
    is_ok = True  # until proven otherwise

    # do some preliminary checks
    if len(bonsai_probe_sync) - len(ephys_probe_sync) > 20 * sampling_rate:
        logger.warning(
            "The sync signals have very different lengths, this cant be!"
        )
        is_ok = False

    if is_ok and abs(np.mean(bonsai_probe_sync) - 2.5) > 1:
        logger.warning(
            "Bonsai signal mean very far from expected average, cant be!"
        )
        is_ok = False

    if is_ok and abs(np.mean(ephys_probe_sync) - 30.0) > 6:
        logger.warning(
            "Ephys signal mean very far from exected average, cant be!"
        )
        is_ok = False

    # get pulses onsets
    bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(
        bonsai_probe_sync, 2.5
    )
    ephys_sync_onsets, ephys_sync_offsets = get_onset_offset(
        ephys_probe_sync, 45
    )

    # remove pulses that are too brief
    errors = np.where(np.diff(bonsai_sync_onsets) < sampling_rate / 3)[0]
    bonsai_sync_offsets = np.delete(bonsai_sync_offsets, errors)
    bonsai_sync_onsets = np.delete(bonsai_sync_onsets, errors)

    # check if numbers make sense
    if len(bonsai_sync_onsets) != len(bonsai_sync_offsets):
        is_ok = False
        logger.warning(
            f"BONSAI - Unequal number of onsets/offsets ({len(bonsai_sync_offsets)}/{len(bonsai_sync_onsets)})"
        )
    if len(ephys_sync_onsets) != len(ephys_sync_offsets):
        is_ok = False
        logger.warning(
            f"EPHYS - Unequal number of onsets/offsets ({len(ephys_sync_offsets)}/{len(ephys_sync_onsets)})"
        )

    # check same results for bonsai and ephys
    if len(bonsai_sync_onsets) != len(ephys_sync_onsets):
        logger.error(
            f"Incosistent number of triggers! Bonsai {len(bonsai_sync_onsets)} and SpikeGLX {len(ephys_sync_onsets)}"
        )
        is_ok = False
        logger.warning(
            "When inspecting probe sync signal found different number of pulses for bonsai "
            f"{len(bonsai_sync_onsets)} and SpikeGLX {len(ephys_sync_onsets)}"
        )
    else:
        logger.debug(
            f"[green]Both bonsai and spikeGLX have {len(ephys_sync_onsets)} sync pulses"
        )

    if ephys_sync_onsets[0] <= bonsai_sync_onsets[0]:
        is_ok = False
        logger.warning("Bonsai should start first!")

    # check the interval between sync signals in bonsai
    onsets_delta = np.diff(bonsai_sync_onsets)
    if len(set(onsets_delta)) > 1:
        counts = {
            k: len(onsets_delta[onsets_delta == k]) for k in set(onsets_delta)
        }
        logger.warning(f"Bonsai sync triggers have variable delay: {counts}")
    elif list(onsets_delta)[0] != sampling_rate:
        # check that it lasts as long as it should
        is_ok = False
        logger.warning(
            f"Bonsai sync triggers are not 1s apart (got {list(onsets_delta)[0]} instead of {sampling_rate})"
        )

    return (
        is_ok,
        bonsai_sync_onsets,
        bonsai_sync_offsets,
        ephys_sync_onsets,
        ephys_sync_offsets,
    )


def validate_recording(
    ai_file_path, ephys_ap_data_path, debug=False, sampling_rate=30000
):
    """
        Checks that an ephys recording and bonsai behavior recording
        are correctly syncd. To do this:
        1. check that number of recording sync signal pulses is the same for both sources

        Arguments:
            ai_file_pat: str. Path to .bin analog inputs file
            ephys_ap_data_path: str. Path to .bin with AP ephys data.
    """

    name = Path(ai_file_path).name
    logger.debug(f"\nRunning validate RECORDING on {name}")

    # load analog from bonsai
    try:
        bonsai_probe_sync = load_or_open(
            ephys_ap_data_path, "bonsai", ai_file_path, 3
        )
    except FileNotFoundError as e:
        # raise FileNotFoundError(e)
        logger.warning(f"Failed to find recording data: {e}")
        return False, 0, 0, f"No rec data found {e}"

    # load data from ephys (from local file if possible)
    ephys_probe_sync = load_or_open(
        ephys_ap_data_path,
        "ephys",
        get_recording_local_copy(ephys_ap_data_path),
        -1,
        order="F",
        dtype="int16",
        nsigs=385,
    )

    # check for aberrant signals in bonsai
    errors = np.where((bonsai_probe_sync < -0.1) & (bonsai_probe_sync > 5.1))[
        0
    ]
    if len(errors):
        logger.info(
            f"Found {len(errors)} samples with too high values in bonsai probe signal"
        )
        if len(errors) > 1000:
            logger.warning(f"This value seems to long, retuirning failure")
            return False, 0, 0, "too_many_errors_in_behavior_sync_signal"
    bonsai_probe_sync[errors] = bonsai_probe_sync[errors - 1]

    # check for aberrant signals in ephys
    errors = np.where(ephys_probe_sync > 75)[0]
    if len(errors):
        logger.warning(
            f"Found {len(errors)} samples with too high values in probe signal"
        )
        if len(errors) > 1000:
            return False, 0, 0, "too_many_errors_in_ephys_sync_signal"
    ephys_probe_sync[errors] = ephys_probe_sync[errors - 1]

    # find probe sync pulses in both
    (
        is_ok,
        bonsai_sync_onsets,
        bonsai_sync_offsets,
        ephys_sync_onsets,
        ephys_sync_offsets,
    ) = get_onsets_offsets(bonsai_probe_sync, ephys_probe_sync, sampling_rate)

    # get time scaling factor
    time_scaling_factor = 1 / (
        (ephys_sync_onsets[-1] - ephys_sync_onsets[0])
        / (bonsai_sync_onsets[-1] - bonsai_sync_onsets[0])
    )

    # debugging plots
    if debug or not is_ok:
        plot_recording_triggers(
            bonsai_probe_sync,
            ephys_probe_sync,
            bonsai_sync_onsets,
            bonsai_sync_offsets,
            ephys_sync_onsets,
            ephys_sync_offsets,
            sampling_rate,
            time_scaling_factor,
        )
        # plt.show()

    return is_ok, ephys_sync_onsets[0], time_scaling_factor, "rec_valid"


if __name__ == "__main__":
    fld = Path("J:\\test_data")

    # ran quality control on test files
    vid = fld / ("FC_210714_AAA1110751_r4_hairpin" + "_video.avi")
    ai = fld / ("FC_210714_AAA1110751_r4_hairpin" + "_analog.bin")
    sampling_rate = 30000

    ephis_ap = fld / (
        "210714_750_longcol_intref_hairpin" + "_g0_t0.imec0.ap.bin"
    )

    # validate_behavior(vid, ai, sampling_rate)
    validate_recording(ai, ephis_ap, debug=True)

    plt.show()
