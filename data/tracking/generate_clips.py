import sys

sys.path.append("./")

from loguru import logger
from pathlib import Path
import numpy as np

from fcutils.path import files
from fcutils import video as video_utils
from fcutils.progress import track

"""
    Generate short clips from the videos we have
"""


dlc_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\dlc")
raw_videos_folder = Path(r"W:\swc\branco\Federico\Locomotion\raw\video")


raw_videos = files(raw_videos_folder, "FC_*.avi")
raw_videos = [
    v
    for v in raw_videos
    if "_d" not in v.name
    and "test" not in v.name.lower()
    and "t_" not in v.name
]
logger.debug(f"Found {len(raw_videos)} videos")


N_clips_per_vid = 4
N_frames_per_vid = 60 * 15

for video in track(raw_videos):
    try:
        n_frames = video_utils.get_video_params(video)[0]
    except ValueError as e:
        logger.warning(
            f"Failed to get data bout video {video.name} with error: {e}"
        )
        continue

    clips_starts = np.random.uniform(
        2000, n_frames - 2000, size=N_clips_per_vid
    ).astype(np.int32)
    for n, start_frame in enumerate(clips_starts):
        save_path = dlc_folder / "videos" / (video.stem + f"_{n}.avi")
        if save_path.exists():
            print("skippy syoppy")
            continue
        logger.info(f'Saving clip {n}: "{save_path.stem}"')

        video_utils.trim_clip(
            video,
            save_path,
            start_frame=start_frame,
            end_frame=start_frame + N_frames_per_vid,
            fps=60,
        )
