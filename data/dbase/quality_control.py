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

        # get frame trigger times
        onsets, offsets = get_onset_offset(analog[:, 0], 4.5)
        return onsets, offsets

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
        frame_trigger_times, frame_trigger_off_times = _get_triggers()
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
                # likely one of the early recordings in which we had 3 channels in the analog binary file
                nsigs = 3
                frame_trigger_times, frame_trigger_off_times = _get_triggers(nsigs=nsigs)
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
    last_frame_s = frame_trigger_off_times[-1] / analog_sampling_rate

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
        frame_trigger_off_times[1],
        "behav_valid",
    )


# ---------------------------------------------------------------------------- #
#                              validate ephys data                             #
# ---------------------------------------------------------------------------- #




def validate_recording(
    ai_file_path,
    ephys_ap_data_path,
    video_duration_s,
    bonsai_first_video_frame,
    bonsai_last_video_frame,
    sampling_rate=30000,
    ephys_sampling_rate=30000,
):
    """
        Checks that an ephys recording and bonsai behavior recording
        are correctly syncd. To do this:
        1. check that number of recording sync signal pulses is the same for both sources

        Arguments:
            ai_file_pat: str. Path to .bin analog inputs file
            ephys_ap_data_path: str. Path to .bin with AP ephys data.
            video_duration_s: float. Duration of video in seconds
            bonsai_first_video_frame: int. First frame of video in bonsai, in samples (trigger onset)
            bonsai_last_video_frame: int. Last frame of video in bonsai, in samples (trigger offset)
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
        return False, 0, 0, 0, f"No rec data found {e}"

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
    logger.info("Loaded BONSAI and EPHYS sync data")

    # check if bonsai_probe_sync starts high
    if bonsai_probe_sync[0] > 2.5:
        bonsai_probe_sync[0] = 0

    # do the same for the ephys
    if ephys_probe_sync[0] > 45:
        ephys_probe_sync[0] = 0

    # get frame triggers for both
    bonsai_sync_onsets, bonsai_sync_offsets = get_onset_offset(
        bonsai_probe_sync, 2.5
    )
    ephys_sync_onsets, ephys_sync_offsets = get_onset_offset(
        ephys_probe_sync, 45
    )

    # check that the first bonsai sync onset is close to the first frame trigger onset
    if np.abs(bonsai_sync_onsets[0] - bonsai_first_video_frame)/sampling_rate > 0.1:
        logger.warning(f"Bonsai sync onset is too far from first frame trigger onset")
        return False, 0, 0, 0, "Bonsai sync onset is too far from first frame trigger onset"

    # check that the last sync offset if slocse to the last frame trigger in bonsai
    if np.abs(bonsai_sync_offsets[-1] - bonsai_last_video_frame)/sampling_rate > 0.1:
        logger.warning(f"Bonsai sync offset is too far from last frame trigger offset")
        return False, 0, 0, 0, "Bonsai sync offset is too far from last frame trigger offset"

    # check that the number of pulses makes sense
    if len(bonsai_sync_onsets) != len(ephys_sync_onsets):
        logger.warning(f"Unequal number of sync pulses, bonsai: {len(bonsai_sync_onsets)} and ephys: {len(ephys_sync_onsets)}")
        return False, 0, 0, 0, f"Different number of sync pulses between bonsai and ephys"
    if len(ephys_sync_onsets) != len(ephys_sync_offsets):
        logger.warning(f"Unequal number of ephys onset/offset, onset: {len(ephys_sync_onsets)} and offset: {len(ephys_sync_offsets)}")
        return False, 0, 0, 0, f"Ephys: different # onset and offsets"
    if len(bonsai_sync_onsets) != len(bonsai_sync_offsets):
        logger.warning(f"Unequal number of bonsai onset/offset, onset: {len(bonsai_sync_onsets)} and offset: {len(bonsai_sync_offsets)}")
        return False, 0, 0, 0, f"Bonsai: different # onset and offsets"

    # check that the number of pulses makes sense given the video duration
    expected_n_pulses = video_duration_s # (there should be a pulse every second)
    if len(bonsai_sync_onsets) != expected_n_pulses:
        logger.warning(f"Expected {expected_n_pulses} based on video duration, but got {len(bonsai_sync_onsets)}")
        return False, 0, 0, 0, f"Wrong number of sync pulses based on video duration"

    # check that the delta-t between pulses is correct
    bonsai_dt = np.diff(bonsai_sync_onsets)
    bonsai_dt_s = np.mean(bonsai_dt)/sampling_rate
    if abs(bonsai_dt_s - 1) > .05:
        logger.warning(f"Bonsai sync pulses are not 1s apart (got {bonsai_dt_s} instead of 1)")
        return False, 0, 0, 0, f"Unexpected bonsai sync pulse spacing"

    ephys_dt = np.diff(ephys_sync_onsets)
    ephys_dt_s = np.mean(ephys_dt)/ephys_sampling_rate
    if abs(ephys_dt_s - 1) > .05:
        logger.warning(f"Ephys sync pulses are not 1s apart (got {ephys_dt_s} instead of 1)")
        return False, 0, 0, 0, f"Unexpected ephys sync pulse spacing"


    # keep the sample number of first onset and last offset for each data stream
    bonsai_first_onset = bonsai_sync_onsets[0]
    bonsai_last_offset = bonsai_sync_offsets[-1]
    ephys_first_onset = ephys_sync_onsets[0]  # ephys keeps 1s before/after recording start/stop so we need to keep track of that
    ephys_last_offset = ephys_sync_offsets[-1]


    # debugging plots
    # if debug or not is_ok:
    #     plot_recording_triggers(
    #         bonsai_probe_sync,
    #         ephys_probe_sync,
    #         bonsai_sync_onsets,
    #         bonsai_sync_offsets,
    #         ephys_sync_onsets,
    #         ephys_sync_offsets,
    #         sampling_rate,
    #         time_scaling_factor,
    #     )
        # plt.show()

    return True, bonsai_first_onset, bonsai_last_offset, ephys_first_onset, ephys_last_offset


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
