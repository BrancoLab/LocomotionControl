from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from rich.logging import RichHandler
from myterial import indigo_light as il


from fcutils.video.utils import get_cap_from_file, get_video_params
from fcutils.maths.stimuli_detection import get_onset_offset


logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)


"""
Take files from bonsai and analyse them to check everything's fine, then run DLC on them

"""


def load_bonsai(folder, name):
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

        Returns:
            video_path: Path. Path to video file
            stimuli: np.ndarray. Array of stim start times in frames wrt video

    """
    logger.debug(
        f"[{il}]Loading bonsai data from folder {folder} with name [b salmon]{name}"
    )
    folder = Path(folder)

    # load analog
    analog = np.fromfile(
        folder / (name + "_analog.bin"), dtype=np.double
    ).reshape(-1, 2)

    # load video metadata
    video = str(folder / (name + "_video.avi"))
    nframes, w, h, fps, _ = get_video_params(get_cap_from_file(video))
    logger.debug(
        f"[{il}]Video params: {nframes} frames {w}x{h}px at {fps} fps | {nframes/fps:.2f}s in total"
    )

    # load stimuli and frame deltas
    diff_array = pd.read_csv(folder / (name + "_camera.csv")).Item2.values
    stimuli = pd.read_csv(folder / (name + "_stimuli.csv")).Item2

    # check that no missing frames occurred
    if np.all(diff_array[1:]) == 1:
        logger.debug(f"[b green]Diff array didnt report any missing frames")

    else:
        logger.warning(f"[b r]found missing frames in diff_array!")

    frame_trigger_times = get_onset_offset(analog[:, 0], 2.5)[0]
    if len(frame_trigger_times) == len(diff_array):
        logger.debug(
            "[b green]Number of trigger times matches number of frames"
        )
    else:
        logger.warning(
            f"[b red]mismatch between frame triggers and number of frames. "
            f"Found {len(frame_trigger_times)} triggers and {len(diff_array)} frames"
        )

    # Get stimuli times in frame number
    # stim_start =
    # TODO extract stim start from files
    # TODO check that the number of stim starts matches what's in the stimuli csv file
    logger.warning("[orange]Stimuli start time extract not yet implemented")

    return Path(video), stimuli


if __name__ == "__main__":
    load_bonsai(
        "Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD_raw",
        "FC_210107_test_longer",
    )
