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

    def _get_triggers(nsigs=4):
        # load analog
        analog = load_bin(ai_file_path, nsigs=nsigs)

        # check that the number of frames is correct
        frame_trigger_times = get_onset_offset(analog[:, 0], 2.5)[0]
        return frame_trigger_times

    name = Path(video_file_path).name
    logger.debug(f"Running validate bonsai on {name}")

    # load video and get metadata
    nframes, w, h, fps, _ = get_video_params(video_file_path)
    if fps != 60:
        raise ValueError("Expected video FPS: 60")

    try:
        frame_trigger_times = _get_triggers()
    except ValueError:
        logger.warning(
            f"While validating bonsay for {name} could not open binary file {ai_file_path}"
        )
        return False, 0

    if len(frame_trigger_times) != nframes:
        try:
            nsigs = 3
            frame_trigger_times = _get_triggers(nsigs=nsigs)
            if len(frame_trigger_times) != nframes:
                raise ValueError
        except ValueError:
            logger.warning(
                f"session: {name} - found {nframes} video frames and {len(frame_trigger_times)} trigger times in analog input"
            )
            return False, nsigs
    else:
        logger.debug(
            f"{name} has {nframes} frames and {len(frame_trigger_times)} trigger times were found"
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

    return True, nsigs


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

    if ephys_sync_onsets[0] <= bonsai_sync_onsets[0]:
        raise ValueError("Bonsai should start first!")

    # check the interval between syn signals in bonsai
    onsets_delta = set(np.diff(bonsai_sync_onsets))
    if len(onsets_delta) > 1:
        raise ValueError(
            f"Bonsai sync triggers have variable delay: {onsets_delta}"
        )
    elif list(onsets_delta)[0] != sampling_rate:
        raise ValueError(
            f"Bonsai sync triggers are not 1s apart (got {list(onsets_delta)[0]} instead of {sampling_rate})"
        )

    # debugging plots
    if debug:
        frames_cut = analog[
            bonsai_sync_onsets[0] : bonsai_sync_onsets[-1] + sampling_rate + 1,
            0,
        ]
        bonsai_cut = bonsai_probe_sync[
            bonsai_sync_onsets[0]
            - 1 : bonsai_sync_onsets[-1]
            + sampling_rate
            + 1
        ]
        ephys_cut = ephys_probe_sync[
            ephys_sync_onsets[0]
            - 1 : ephys_sync_onsets[-1]
            + sampling_rate
            + 1
        ]

        f, axes = plt.subplots(ncols=2)
        axes[0].plot(
            frames_cut[: 20 * sampling_rate] * 0.5,
            color="b",
            label="bonsai frames",
        )
        axes[0].plot(
            bonsai_cut[: 20 * sampling_rate], color="k", label="bonsai"
        )
        axes[0].plot(
            ephys_cut[: 20 * sampling_rate] / 10, color="g", label="ephys"
        )
        axes[0].legend()

        axes[1].plot(
            frames_cut[-40 * sampling_rate :] * 0.5,
            color="b",
            label="bonsai frames",
        )
        axes[1].plot(
            bonsai_cut[-40 * sampling_rate :], color="k", label="bonsai"
        )
        axes[1].plot(
            ephys_cut[-40 * sampling_rate :] / 10, color="g", label="ephys"
        )
        axes[1].legend()

        logger.debug("Showing plot")
        plt.show()

    return (
        bonsai_sync_onsets[0],
        bonsai_sync_onsets[-1] + sampling_rate,
        ephys_sync_onsets[-1],
        ephys_sync_onsets[-1] + sampling_rate,
    )


if __name__ == "__main__":
    fld = Path("J:\\test_data")

    # ran quality control on test files
    vid = fld / "FC_210505_AAA111075_recording_test_run_video.avi"
    ai = fld / "FC_210505_AAA111075_recording_test_run_analog.bin"
    sampling_rate = 30000

    ephis_ap = fld / "test_run_g0_t0.imec0.ap.bin"

    # validate_bonsai(vid, ai, sampling_rate)
    validate_recording(ai, ephis_ap, debug=True)
