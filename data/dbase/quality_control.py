import numpy as np
from pathlib import Path
from loguru import logger
import sys

sys.path.append("./")

import matplotlib.pyplot as plt
from myterial import blue_dark, salmon_dark, indigo

from fcutils.video import get_video_params
from fcutils.maths.signals import get_onset_offset

from data.dbase._tables import load_bin


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
        nsigs = 4
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


# ---------------------------------------------------------------------------- #
#                              validate ephys data                             #
# ---------------------------------------------------------------------------- #
def plot_recording_triggers(
    bonsai_sync,
    ephys_sync,
    bonsai_sync_onsets,
    bonsai_sync_offsets,
    ephys_sync_onsets,
    ephys_sync_offsets,
    sampling_rate,
    time_scaling_factor,
):
    N = 50
    bonsai_time = (
        np.arange(0, len(bonsai_sync), step=N) - bonsai_sync_onsets[0]
    ) / sampling_rate
    probe_time = (
        (np.arange(0, len(ephys_sync), step=N) - ephys_sync_onsets[0])
        / sampling_rate
        * time_scaling_factor
    )

    # plot sync signals
    f, axes = plt.subplots(
        figsize=(16, 8), ncols=2, gridspec_kw={"width_ratios": [3, 1]}
    )
    axes[0].plot(bonsai_time, bonsai_sync[::N] / 5, lw=0.5, color="b")
    axes[0].plot(probe_time, ephys_sync[::N] * 0.02, lw=0.5, color="salmon")

    # mark trigger onsets
    bad_onsets = np.where(np.diff(bonsai_sync_onsets) != sampling_rate)[0]
    probe_bad_onsets = np.where(
        np.abs(np.diff(ephys_sync_onsets) - sampling_rate) > sampling_rate / 2
    )[0]

    axes[0].scatter(
        (bonsai_sync_onsets[bad_onsets] - bonsai_sync_onsets[0])
        / sampling_rate,
        np.ones_like(bad_onsets) * 1.4,
        marker="v",
        color=indigo,
        alpha=0.75,
        s=75,
        zorder=120,
    )
    axes[0].scatter(
        (bonsai_sync_onsets - bonsai_sync_onsets[0]) / sampling_rate,
        np.ones_like(bonsai_sync_onsets) * 1.2,
        marker="v",
        color=blue_dark,
        zorder=100,
    )

    axes[0].scatter(
        (ephys_sync_onsets - ephys_sync_onsets[0])
        / sampling_rate
        * time_scaling_factor,
        np.ones_like(ephys_sync_onsets) * 1.25,
        marker="v",
        color=salmon_dark,
        zorder=100,
    )
    axes[0].scatter(
        (ephys_sync_onsets[probe_bad_onsets] - ephys_sync_onsets[0])
        / sampling_rate
        * time_scaling_factor,
        np.ones_like(probe_bad_onsets) * 1.45,
        marker="v",
        s=85,
        alpha=0.7,
        color=salmon_dark,
        zorder=100,
    )

    axes[0].set(title="blue: bonsai | salmon: probe", xlabel="time (s)")

    # plot histogram of triggers durations
    axes[1].hist(np.diff(bonsai_sync_onsets), bins=50, color="b", alpha=0.5)
    axes[1].hist(
        np.diff(ephys_sync_onsets), bins=50, color="salmon", alpha=0.5
    )
    axes[1].set(title="blue: bonsai | salmon: probe", xlabel="offset duration")


def get_onsets_offsets(bonsai_probe_sync, ephys_probe_sync, sampling_rate):
    logger.debug("extracting sync signal pulses")
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
        raise ValueError(
            f"BONSAI - Unequal number of onsets/offsets ({len(bonsai_sync_offsets)}/{len(bonsai_sync_onsets)})"
        )
    if len(ephys_sync_onsets) != len(ephys_sync_offsets):
        raise ValueError(
            f"EPHYS - Unequal number of onsets/offsets ({len(ephys_sync_offsets)}/{len(ephys_sync_onsets)})"
        )

    # check same results for bonsai and ephys
    if len(bonsai_sync_onsets) != len(ephys_sync_onsets):
        logger.error(
            f"Incosistent number of triggers! Bonsai {len(bonsai_sync_onsets)} and SpikeGLX {len(ephys_sync_onsets)}"
        )
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

    # check the interval between sync signals in bonsai
    onsets_delta = np.diff(bonsai_sync_onsets)
    if len(set(onsets_delta)) > 1:
        counts = {
            k: len(onsets_delta[onsets_delta == k]) for k in set(onsets_delta)
        }
        logger.warning(f"Bonsai sync triggers have variable delay: {counts}")
    elif list(onsets_delta)[0] != sampling_rate:
        # check that it lasts as long as it should
        raise ValueError(
            f"Bonsai sync triggers are not 1s apart (got {list(onsets_delta)[0]} instead of {sampling_rate})"
        )

    return (
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
    logger.debug(f"\nRunning validate recordings on {name}")

    # load analog from bonsai
    analog = load_bin(ai_file_path, nsigs=4)
    bonsai_probe_sync = analog[:, 3].copy()

    # load data from ephys
    ephys = load_bin(ephys_ap_data_path, nsigs=385, order="F", dtype="int16")
    ephys_probe_sync = ephys[:, -1].copy()

    # check for aberrant signals in bonsai
    errors = np.where((bonsai_probe_sync != 0) & (bonsai_probe_sync != 1))[0]
    if len(errors):
        logger.warning(
            f"Found {len(errors)} samples with too high values in probe signal"
        )
    bonsai_probe_sync[errors] = bonsai_probe_sync[errors - 1]

    # check for aberrant signals in ephys
    errors = np.where(ephys_probe_sync > 70)[0]
    if len(errors):
        logger.warning(
            f"Found {len(errors)} samples with too high values in probe signal"
        )
    ephys_probe_sync[errors] = ephys_probe_sync[errors - 1]

    # close stuff to save memory
    logger.debug(
        f"{name}  | Bonsai analogs shape: {analog.shape} | ephys data shape: {ephys.shape}"
    )
    del ephys
    del analog

    # find probe sync pulses in both
    (
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
    if debug:
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
