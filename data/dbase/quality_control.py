import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from fcutils.video import get_video_params
from fcutils.maths.signals import get_onset_offset


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

    # load video and get metadata
    nframes, w, h, fps, _ = get_video_params(video_file_path)
    if fps != 60:
        raise ValueError("Expected video FPS: 60")

    # load analog
    analog = np.fromfile(ai_file_path, dtype=np.double).reshape(
        -1, 4  # 4 is the number of analog inputs recorded in bonsai
    )

    # check that the number of frames is correct
    frame_trigger_times = get_onset_offset(analog[:, 0], 2.5)[0]
    if len(frame_trigger_times) != nframes:
        raise ValueError(
            f"session: {name} - found {nframes} video frames and {frame_trigger_times} trigger times in analog input"
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

    return True


def validate_recording(ai_file_path, ephys_ap_data_path, plot=False):
    """
        Checks that an ephys recording and bonsai behavior recording
        are correctly syncd. To do this:
        1. cut the ephys recording to remove 'context' before and after recording
        2. check that number of recording sync signal pulses is the same for both sources

        Arguments:
            ai_file_pat: str. Path to .bin analog inputs file
            ephys_ap_data_path: str. Path to .bin with AP ephys data.
    """
    # load analog from bonsai
    analog = np.fromfile(ai_file_path, dtype=np.double).reshape(
        -1, 4  # 4 is the number of analog inputs recorded in bonsai
    )
    bonsai_probe_sync = analog[:, 3]

    # load data from ephys
    ephys = np.fromfile(ephys_ap_data_path, dtype=np.double).rehsape(-1, 384)
    ephys_probe_sync = ephys[:, -1]

    # debugging plots
    if plot:
        f, ax = plt.subplots()
        ax.plot(bonsai_probe_sync, color="k", label="bonsai")
        ax.plot(ephys_probe_sync, color="g", label="ephys")

        plt.show()


if __name__ == "__main__":
    # ran quality control on test files
    vid = ""
    ai = ""
    sampling_rate = 30000

    ephis_ap = ""
    ephis_ap_meta = ""

    validate_bonsai(vid, ai, sampling_rate)
