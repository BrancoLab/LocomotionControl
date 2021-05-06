import numpy as np
from pathlib import Path
from loguru import logger
import sys

sys.path.append("./")

import matplotlib.pyplot as plt

from fcutils.video import get_video_params
from fcutils.maths.signals import get_onset_offset

from data.dbase._tables import load_bin


def validate_bonsai(video_file_path, ai_file_path, analog_sampling_rate):
    """
        Checks that a bonsai behaviour recording ran correctly
        (i.e. no frames dropped)

        Arguments:
            video_file_pat: str. Path to .avi video file
            ai_file_pat: str. Path to .bin analog inputs file
            analog_sampling_rate: int. Sampling rate in bonsai
    """
    name = Path(video_file_path).name
    logger.debug(f"Running validate bonsai on {name}")

    # load video and get metadata
    nframes, w, h, fps, _ = get_video_params(video_file_path)
    if fps != 60:
        raise ValueError("Expected video FPS: 60")

    # load analog
    analog = load_bin(ai_file_path, nsigs=4)

    # check that the number of frames is correct
    frame_trigger_times = get_onset_offset(analog[:, 0], 2.5)[0]
    if len(frame_trigger_times) != nframes:
        raise ValueError(
            f"session: {name} - found {nframes} video frames and {len(frame_trigger_times)} trigger times in analog input"
        )
    logger.debug(
        f"{name} has {nframes} frames and {frame_trigger_times} trigger times were found"
    )

    # check that the number of frames is what you'd expect given the duration of the exp
    first_frame_s = frame_trigger_times[0] / analog_sampling_rate
    last_frame_s = frame_trigger_times[-1] / analog_sampling_rate

    exp_dur = last_frame_s - first_frame_s  # video duration in seconds
    expected_n_frames = np.floor(exp_dur * 60).astype(np.int64)
    if np.abs(expected_n_frames - nframes) > 5:
        raise ValueError(
            f"[b yellow]Expected {expected_n_frames} frames but found {nframes} in video"
        )
    logger.debug(f"{name} video duration is correct")

    return True


def validate_recording(ai_file_path, ephys_ap_data_path, debug=False):
    """
        Checks that an ephys recording and bonsai behavior recording
        are correctly syncd. To do this:
        1. check that number of recording sync signal pulses is the same for both sources

        Arguments:
            ai_file_pat: str. Path to .bin analog inputs file
            ephys_ap_data_path: str. Path to .bin with AP ephys data.
    """
    name = Path(ai_file_path).name
    logger.debug(f"Running validate recordings on {name}")

    # load analog from bonsai
    analog = load_bin(ai_file_path, nsigs=4)
    bonsai_probe_sync = analog[:, 3]

    # load data from ephys
    ephys = load_bin(ephys_ap_data_path, nsigs=385, order="F", dtype="int16")
    ephys_probe_sync = ephys[:, -1]
    logger.debug(
        f"{name}  | Bonsai analogs shape: {analog.shape} | ephys data shape: {ephys.shape}"
    )

    # find probe syn pulses in both
    bonsai_sync_onsets = get_onset_offset(bonsai_probe_sync, 2.5)[0]
    ephys_sync_onsets = get_onset_offset(ephys_probe_sync, 30)[0]

    if len(bonsai_sync_onsets) != len(ephys_sync_onsets):
        raise ValueError(
            "When inspecting probe sync signal found different number of pulses for bonsai "
            f"{len(bonsai_sync_onsets)} and SpikeGLX {len(ephys_sync_onsets)}"
        )
    else:
        logger.debug(
            f"[green]Both bonsai and spikeGLX have {len(ephys_sync_onsets)} sync pulses"
        )

    # debugging plots
    if debug:
        f, ax = plt.subplots()
        ax.plot(analog[:200000, 0] * 0.5, color="k", label="bonsai frames")
        ax.plot(bonsai_probe_sync[:200000], color="k", label="bonsai")
        ax.plot(ephys_probe_sync[:200000] / 10, color="g", label="ephys")
        ax.legend()

        logger.debug("Showing plot")
        plt.show()


if __name__ == "__main__":
    fld = Path("J:\\test_data")

    # ran quality control on test files
    vid = fld / "FC_210505_AAA111075_recording_test_run_video.avi"
    ai = fld / "FC_210505_AAA111075_recording_test_run_analog.bin"
    sampling_rate = 30000

    ephis_ap = fld / "test_run_g0_t0.imec0.ap.bin"

    # validate_bonsai(vid, ai, sampling_rate)
    validate_recording(ai, ephis_ap, debug=True)
