# %%
import sys
sys.path.append("./")

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


from fcutils import video as video_utils
from fcutils.maths.signals import get_onset_offset

from data.dbase.io import load_bin

'''
Given the AI and video for an opto experiment it
    1. detects when the opto stim are delivered 
    2. makes a video with short clips around each stim
    3. if tracking data is provided in plots speed traces aligned to stim onset
'''

# %%
# Load data
folder = Path(r'W:\swc\branco\Federico\Locomotion\raw\optto_test')
exp_name = "FC_211013_BAA110520_hairpin_opto_allrois_frames_test"
analog = load_bin(folder / f'{exp_name}_analog.bin')

video = folder / f'{exp_name}_video.avi'

# %%


video_frames = get_onset_offset(analog[:, 0], 2.5)[0]
print(f'Found {len(video_frames)} video triggers')

nframes, width, height, fps, is_color = video_utils.get_video_params(video)
print(f"Video has: {nframes} frames")

# %%
# Get stim _starts

up = np.where(analog[:, -1] > .1)[0] 
starts = np.array([up[x+1] for x in np.where(np.diff(up) > 30000)[0] ])

first_video_trigger = np.where(analog[:, 0] > 0.5)[0][0]

# %%
f, ax = plt.subplots(figsize=(19, 6))

ax.plot(analog[::10, 0], color='k')
ax.plot(analog[::10, -1])
ax.scatter(starts/10, np.ones_like(starts), zorder=100, color='salmon')
ax.scatter(video_frames/10, np.ones_like(video_frames) * 2, zorder=100, color='red')
ax.set(xlim=[len(analog)/10 - 8000, len(analog)/10])

# %%

font = cv2.FONT_HERSHEY_SIMPLEX

CLEAN = False
FPS = 20


# create an opencv writer
logger.info(f'Opening writer')
save_path = str(folder / f'{video.stem}_clips{"" if not CLEAN else "_clean"}_{FPS}fps.mp4')
writer = video_utils.open_cvwriter(
    save_path,
    w=width+20,
    h=height+20,
    framerate=FPS,
    format=".mp4",
    iscolor=True,
)


logger.info(f'Starting')
videocap = video_utils.get_cap_from_file(video)
n_frames_before, n_frames_after = 60, 180
# iterate over trigger times
for i, t_sample in enumerate(starts):

    trigger = int((t_sample - first_video_trigger) / 30000 * 60)

    # get laser power
    # power = round(2 * np.mean(analog[t_sample:t_sample+30000, -1]), 2)

    # get frames range
    logger.info(f'Adding trigger: {trigger}')
    start = trigger - n_frames_before
    end = trigger + n_frames_after

    if start < 1 or end > nframes:
        raise(ValueError)
        continue
    
    # iterate over frames
    video_utils.cap_set_frame(videocap, start)
    for framen in range(start, end):
        
        ret, frame = videocap.read()
        if not ret:
            logger.debug(
                f"FCUTILS: failed to read frame {framen} while trimming clip [{nframes} frames long]."
            )
            break

        # add marker to show stim is on
        if not CLEAN:
            power = analog[int(framen * 30000 / 60 + first_video_trigger), -1]

            frame = cv2.circle(frame, (50, 50), 25, (0, 0, 0), -1)
            if power > .2:
                frame = cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)

            # make border

            if framen >= trigger-1:
                border_color = (0, 0, 255)
            else:
                border_color = (0, 0, 0)

            frame = cv2.copyMakeBorder(
                frame,
                top=10,
                bottom=10,
                left=10,
                right=10,
                borderType=cv2.BORDER_CONSTANT,
                value=border_color
            )

            # # add text with metadata
            # cv2.putText(frame, f'power: {power}', (10, height - 20), font, 3, (0, 0, 0), 4, cv2.LINE_AA)
            # cv2.putText(frame, f'power: {power}', (10, height - 20), font, 3, (255, 50, 50), 2, cv2.LINE_AA)

        # write
        writer.write(frame)

    # add some black frames
    if not CLEAN:
        for i in range(10):
            writer.write(np.zeros_like(frame))


writer.release()
print('done')



# %%
f, ax = plt.subplots(figsize=(19, 6))
ax.plot(analog[::10, 0][:1000])