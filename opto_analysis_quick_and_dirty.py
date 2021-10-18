# %%

import sys
sys.path.append("./")

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import random 

from data.dbase.io import load_bin, load_dlc_tracking
from data import data_utils
from data.dbase.hairpin_trace import HairpinTrace
from data.dbase._tracking import register

from fcutils.maths.geometry import (
    calc_distance_between_points_in_a_vector_2d as get_speed_from_xy,
)
from fcutils import video as video_utils
from fcutils.maths.signals import get_onset_offset
from fcutils.plot.elements import plot_mean_and_error
from fcutils.maths import derivative

'''
    Plot the speed following opto sitmulation

'''
# %%
# Load data
folder = Path(r'W:\swc\branco\Federico\Locomotion\raw\optto_test')
exp_name = "FC_211013_BAA110521_hairpin_opto_allrois"

# get stim starts
analog = load_bin(folder / f'{exp_name}_analog.bin')
up = np.where(analog[:, -1] > .1)[0] 
starts = np.array([up[x+1] for x in np.where(np.diff(up) > 30000)[0] ]) - 1

# convert to frames
first_video_trigger = np.where(analog[:, 0] > 0.5)[0][0]
starts_frames = [int((x-first_video_trigger)/30000*60) for x in starts]

# get video file
video = folder / f'{exp_name}_video.avi'


# get DLC data
tracking_file = folder / f'{exp_name}_videoDLC_resnet101_locomotionSep13shuffle1_450000.h5'
tracking = load_dlc_tracking(tracking_file)


# get video params
video_frames = get_onset_offset(analog[:, 0], 2.5)[0]
print(f'Found {len(video_frames)} video triggers')

nframes, width, height, fps, is_color = video_utils.get_video_params(video)
print(f"Video has: {nframes} frames")


# %%
ccm = np.load(r'W:\swc\branco\Federico\Locomotion\raw\CMM_matrices\FC_210323_AAA1110752_d1_video.npy')
# Clean DLC data
x, y = tracking['body']['x'].copy(), tracking['body']['y'].copy()

# register to CMM
x, y = register(x, y, ccm)

# scale to go px -> cm
cm_per_px = 60 / 830
x *= cm_per_px
y *= cm_per_px

x -= np.nanmin(x) + 5
y -= np.nanmin(y) + 5
x = 20 - x + 20

# remove low confidence intervals
like = tracking['body']["likelihood"]
x[like < 0.9] = np.nan
y[like < 0.9] = np.nan

# speed = get_speed_from_xy(x, y) * 60
speed = (
    data_utils.convolve_with_gaussian(
        get_speed_from_xy(x, y), kernel_width=2
    )
    * 60
)  # speed in cm / s
speed[:5] = speed[6]
speed[-5:] = speed[-6]

# assign tracking to hairpin global coordinates
segment, gcoord = HairpinTrace().assign_tracking(x, y)


f, ax = plt.subplots(figsize=(8, 14))
ax.plot(x, y, color=[.6, .6, .6], lw=.5)

_ = ax.scatter(x[::10], y[::10], c=gcoord[::10], zorder=100)

# %%
'''
Plot the globalcoorinate values at each stim, speed, tracking for each ROI
'''
gcoord_deriv = derivative(gcoord)

f = plt.figure(figsize=(22, 14))
axes = f.subplot_mosaic(
            """
            AABBB
            AACCC
            AADDD
            AAEEE
            AAFFF
            AAGGG
            """
        )
roi_axes ='BCDEFG'
axes['A'].plot(x, y, color=[.6, .6, .6], lw=.5)

colors = {roi:col for col, roi in zip(('salmon', 'b', 'g', 'red', 'm', 'k'), np.arange(6))}
stim_traces = {roi:[] for roi in np.arange(6)}
random_traces = {roi:[] for roi in np.arange(6)}
n_pre, n_post = 1*60, int(2.5*60)
for n, frame in enumerate(starts_frames):
    frame -= 16

    if gcoord[frame] < .2:
        roi = 0
    elif .24 <= gcoord[frame] < .28:
        roi = 1
    elif .28 <= gcoord[frame] < .40:
        roi = 2
    elif .48 <= gcoord[frame] < .55:
        roi = 3
    elif .6 <= gcoord[frame] < .85:
        roi = 4
    else:
        roi = 5
    color = colors[roi]

    if gcoord[frame - n_pre] > gcoord[frame + n_pre]:
        continue  # keep only outbound 

    # plot position on XY tracking
    axes['A'].scatter(x[frame], y[frame], color=color, zorder=100, s=150)

    if roi is not None:
        # axes[roi_axes[roi]].plot(speed[frame - n_pre : frame + n_post], color=color, alpha=.5)
        stim_traces[roi].append(speed[frame - n_pre : frame + n_post])


    shift = 20
    for random_frame in np.where(
                (abs(gcoord - gcoord[frame - shift])<.05) & 
                # (abs(speed - speed[frame-30]) < 2) & 
                (abs(speed - speed[frame - shift]) < 2) & 
                (speed > 35) & 
                # (abs(speed - speed[frame]) < 5) & 
                (abs(gcoord - gcoord[frame]) < 0.02) &
                (abs(gcoord_deriv - gcoord_deriv[frame]) < 0.01) & 
                # (gcoord >= gcoord[frame]) & 
                (gcoord_deriv >= 0)
                )[0][:10]:

        if abs(random_frame - frame) < 60:
            continue

        if random_frame > n_pre and random_frame < len(x) - n_post:
            axes['A'].scatter(x[random_frame], y[random_frame], ec='k', lw=1, color=color, zorder=200)
            random_traces[roi].append(speed[random_frame - n_pre : random_frame + n_post])

# plot mean of stim traces
for roi, traces in stim_traces.items():
    if traces:
        plot_mean_and_error(
            np.nanmean(np.vstack(traces), axis=0), 
            np.nanstd(np.vstack(traces), axis=0), 
            axes[roi_axes[roi]],
            lw=6, color=colors[roi], zorder=40
        )

# plot mean of random traces
for roi, traces in random_traces.items():
    if traces:
        plot_mean_and_error(
            np.nanmean(np.vstack(traces), axis=0), 
            np.nanstd(np.vstack(traces), axis=0), 
            axes[roi_axes[roi]],
            lw=6, color=[.3, .3, .3], zorder=20
        )

for ax in roi_axes:
    axes[ax].axvline(n_pre, lw=4, ls='--', color='k')
    axes[ax].set(ylim=[0, 80])


# %%
