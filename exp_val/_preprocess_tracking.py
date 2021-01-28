from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from rich.logging import RichHandler
from myterial import indigo_light as il


from fcutils.video.utils import get_cap_from_file, get_video_params
from fcutils.maths.stimuli_detection import get_onset_offset
from fcutils.maths.utils import derivative


logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)


TRACKING_BASH_TEMPLATE = """#! /bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 80G # memory pool for all cores
#SBATCH --gres=gpu:gtx1080:1
#SBATCH -n 10
#SBATCH -t 2-0:0 # time
#SBATCH	-o err.err
#SBATCH -e err.err
#SBATCH --mail-user=federicoclaudi@protonmail.com
#SBATCH --mail-type=FAIL

echo "loading conda env"
module load miniconda
module load nvidia/9.0

conda activate dlc
export DLClight=True
export CUDA_VISIBLE_DEVICES=1

echo "running tracking"
python /nfs/winstor/branco/Federico/Locomotion/control/LocomotionControl/exp_val/dlc_on_hpc.py \\
        /nfs/winstor/branco/Federico/Locomotion/control/experimental_validation/2WDD/Kinematics_FC-FC-2021-01-25/config.yaml \\
        VIDEO \\
        SAVE
"""


def make_bash_text(experiment, video_path, save_path):
    """
        Creates a string with the content of a .sh script for running 
        deeplabcut on HPC
    """
    if not Path(video_path).exists():
        raise ValueError("Video doesnt exist")
    if not Path(save_path).exists():
        raise ValueError("Save folder doesnt exist")

    video_path = video_path.replace("Z:\\swc\\", "/nfs/winstor/").replace(
        "\\", "/"
    )
    save_path = save_path.replace("Z:\\swc\\", "/nfs/winstor/").replace(
        "\\", "/"
    )

    return (
        TRACKING_BASH_TEMPLATE.replace("EXP", experiment)
        .replace("VIDEO", video_path)
        .replace("SAVE", save_path)
    )


def load_bonsai(folder, name, exp_fps):
    """
        Load all data saved by a bonsai session (excluding the video, for that it only gets metadata)
        and checks that everything's okay (e.g. no dropped frames). The data loaded include:
        * analog: Nx2 (N=n samples) numpy array with camera triggers and audio stimuli
        * diff_array stores if frames were acquired succesfully
        * stimuli stores the timestamp of the start of each stimulus
        This function also takes care of converting stimuli times from timestamps to 
        frame number


        Arguments:
            folder: str, Path. Path to folder with data
            name: str. Name of session 
            exp_fps: int. Expect fps of video

        Returns:
            video_path: Path. Path to video file
            stimuli: np.ndarray. Array of stim start times in frames wrt video

    """
    analog_sampling_rate = 30000

    logger.debug(
        f"[{il}]Loading bonsai data from folder {folder} with name [b salmon]{name}"
    )
    folder = Path(folder)

    # load analog
    logger.debug("loading analog")
    analog = np.fromfile(
        folder / (name + "_analog.bin"), dtype=np.double
    ).reshape(-1, 2)

    # load video metadata
    logger.debug("loading video")
    video = str(folder / (name + "_video.avi"))
    nframes, w, h, fps, _ = get_video_params(get_cap_from_file(video))
    logger.debug(
        f"[{il}]Video params: {nframes} frames {w}x{h}px at {fps} fps | {nframes/fps:.2f}s in total"
    )
    if fps != exp_fps:
        logger.warning(
            f"Video fps: {fps} is different from expected experiment fps: {exp_fps}"
        )

    # load stimuli and frame deltas
    logger.debug("loading stimuli")
    diff_array = pd.read_csv(folder / (name + "_camera.csv")).Item2.values
    stimuli = pd.read_csv(folder / (name + "_stimuli.csv")).Item2

    # check that no missing frames occurred
    if np.all(diff_array[1:]) == 1:
        logger.debug(f"[b green]Diff array didnt report any missing frames")
    else:
        logger.warning(f"[b r]found missing frames in diff_array!")

    frame_trigger_times = get_onset_offset(analog[:, 0], 2.5)[0]
    if len(frame_trigger_times) == nframes:
        logger.debug(
            "[b green]Number of trigger times matches number of frames"
        )
    else:
        logger.warning(
            f"[b red]mismatch between frame triggers and number of frames. "
            f"Found {len(frame_trigger_times)} triggers and {nframes} frames"
        )
        if np.abs(len(frame_trigger_times) - nframes) > 100:
            raise ValueError(
                f"[b red]mismatch between frame triggers and number of frames. "
                f"Found {len(frame_trigger_times)} triggers and {nframes} frames"
            )

        """ tp plot stuff for inspection
            import matplotlib.pyplot as plt
            tms = frame_trigger_times[frame_trigger_times < 150000]
            plt.plot(analog[:150000, 0])
            plt.scatter(
                tms,
                np.ones_like(tms),
                color="r",
                s=200,
                zorder=100,
            )
            plt.show()
        """

    # check that the number of frames is what you'd expect given the duration of the exp
    logger.info(
        f"Experiment duration: {int(len(analog) / analog_sampling_rate / 60)} minutes"
    )
    first_frame_s = frame_trigger_times[0] / analog_sampling_rate
    last_frame_s = frame_trigger_times[-1] / analog_sampling_rate
    exp_dur = last_frame_s - first_frame_s  # video duration in seconds
    expected_n_frames = np.floor(exp_dur * exp_fps).astype(np.int64)
    if np.abs(expected_n_frames - nframes) > 5:
        logger.warning(
            f"[b yellow]Expected {expected_n_frames} frames but found {nframes} in video"
        )
    else:
        logger.debug(
            "[b green]Number of frames found matches what youd expect from video duration"
        )

    # Get stimuli times in sample number
    stim_starts = get_onset_offset(analog[:, 1], 1.5)[0]
    later_starts = stim_starts[derivative(stim_starts) > 1000]
    if len(later_starts):
        stim_starts = np.concatenate([[stim_starts[0]], later_starts])
    else:
        stim_starts = np.array(stim_starts[0]).reshape(1)

    if not stimuli.empty:
        if len(stim_starts) == len(stimuli):
            logger.debug("[b green]Number of stimuli starts is correct")
        else:
            logger.warning(
                f"[b red]Expected: {len(stimuli)} stimuli but found {len(stim_starts)} detect stimuli onsets"
            )

        # get stimuli times in frame number
        stim_frames = []
        for stim in stim_starts:
            stim_frames.append(np.abs(frame_trigger_times - stim).argmin())
        stim_frames = np.array(stim_frames)
        logger.debug(f"Found {len(stim_frames)} stimuli")

        if np.any(stim_frames > nframes):
            raise ValueError(
                "Found a stimulus that appears to have happened after the first video frame"
            )
    else:
        logger.info("No stimuli found in experiment")
        stim_frames = np.array([])

    return Path(video), stim_frames


if __name__ == "__main__":
    fld = "Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD\\RAW"
    name = "FC_210128_BA1099282"
    load_bonsai(fld, name, 60)
