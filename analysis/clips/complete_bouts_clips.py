# %%
import sys
sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
import cv2
import time

from fcutils.progress import track
from fcutils import video as video_utils

from data.dbase import db_tables


'''
    Creates video clips of each complete bout, saving one video per experiment.
'''

save_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\bouts_clips")


# get bouts
bouts = pd.DataFrame(
    (
        db_tables.LocomotionBouts & 'complete="true"' & 'direction="outbound"' & 'duration<8'
    ).fetch()
)
logger.info(f"Found {len(bouts)} complete bouts")

# get tracking for each session
sessions = bouts["name"].unique()

# TODO overlay tracking when present


# %%
font = cv2.FONT_HERSHEY_SIMPLEX
FPS = 5

for session in track(sessions):
    # get session bout
    session_bouts = bouts.loc[bouts["name"] == session]
    save_path = save_folder / (session + f'_{len(session_bouts)}_bouts_clips_{FPS}fps.mp4')

    # open video writer
    video = (db_tables.Session & f'name="{session}"').fetch1("video_file_path")
    videocap = video_utils.get_cap_from_file(video)
    _, width, height, _, _ = video_utils.get_video_params(video)
    writer = video_utils.open_cvwriter(
        str(save_path),
        w=width,
        h=height,
        framerate=FPS,
        format=".mp4",
        iscolor=True,
    )

    # save each bout
    for i, bout in session_bouts.iterrows():

        video_utils.cap_set_frame(videocap, bout.start_frame - 10)
        for frame in range(bout.start_frame - 10, bout.end_frame + 10):
            ret, frame = videocap.read()
            if not ret:
                raise ValueError(f"Failed {video} to open video at frame {frame}")

            # write
            writer.write(frame)

        # add separation between the two
        # for i in range(10):
        #     blank = np.zeros_like(frame)
        #     cv2.putText(blank, f'playing bout {i}', (100, height - 200), font, 3, (255, 50, 50), 2, cv2.LINE_AA)
        #     writer.write(blank)


    writer.release()
    