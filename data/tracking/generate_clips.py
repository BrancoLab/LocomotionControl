import sys

sys.path.append("./")

from loguru import logger
from pathlib import Path
import numpy as np
from random import choices

from fcutils.path import files
from fcutils import video as video_utils
from fcutils.progress import track


"""
    Generate short clips from the videos we have
"""


from data.dbase.db_tables import Tracking

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

    # get when mouse is going fast
    tracking = Tracking.get_session_tracking(video.stem.split('_video')[0], body_only=True, movement=True)

    if tracking.empty:
        print('Skippy because no tracking')
        continue

    clips_starts = sorted(choices(np.where(tracking.walking ==1)[0], k=N_clips_per_vid))
    for n, start_frame in enumerate(clips_starts):
        save_path = dlc_folder / "videos" / (video.stem + f"_{n}.avi")
        if save_path.exists():
            print("skippy stoppy")
            continue
        logger.info(f'Saving clip {n}: "{save_path.stem}"')

        video_utils.trim_clip(
            video,
            save_path,
            start_frame=start_frame,
            end_frame=start_frame + N_frames_per_vid,
            fps=60,
        )
