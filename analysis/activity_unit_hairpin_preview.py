import sys
from tpd import recorder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from loguru import logger

sys.path.append("./")
from pathlib import Path

from fcutils.plot.figure import clean_axes
from myterial.utils import make_palette
from myterial import grey_darker, grey_light,amber_light, amber_darker, grey_dark, blue_grey

from analysis import visuals
from analysis._visuals import move_figure

from data.dbase import db_tables
from data import data_utils, colors

base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")

"""
    Plot a few exploratory plots for each session a recording was performed on.

        1. Overview of tracking data  
"""



sessions = (
    db_tables.ValidatedSession * db_tables.Session
    & "is_recording=1"
    & 'arena="hairpin"'
)


for session in sessions:
    # start a new recorder sesssion
    recorder.start(
        base_folder=base_folder, name=session["name"], timestamp=False
    )

    # load tracking data
    body_tracking = db_tables.Tracking.get_session_tracking(
        session["name"], body_only=True
    )
    frames =np.arange(0, len(body_tracking.x), 60*60*5)
    time = (frames / 60 / 60).astype(np.int32)

    downsampled_tracking = db_tables.Tracking.get_session_tracking(
        session["name"], body_only=True
    )
    data_utils.downsample_tracking_data(downsampled_tracking, factor=10)

    # get locomotion bouts
    bouts = db_tables.LocomotionBouts.get_session_bouts(session['name'])
    out_bouts = bouts.loc[bouts.direction=='outbound']
    in_bouts = bouts.loc[bouts.direction=='inbound']

    # get spikes data
    units = db_tables.Unit.get_session_units(session['name'], spikes=True)

    # get tone onset times
    tone_onsets = db_tables.Behavior.get_session_onsets(session['name'], 'speaker')
    logger.debug(f'Found {len(tone_onsets)} tones in session')

    for i, unit in units.iterrows():
        # get tracking data at each spike
        unit_tracking = data_utils.select_by_indices(body_tracking, unit.spikes)

        # get unit spike raster
        body_tracking['unit_spikes_raster'] = db_tables.Unit.get_unit_raster(session['name'], unit['unit_id'])

        # crate figure
        f = plt.figure(figsize=(24, 12))
        axes = f.subplot_mosaic(
            """
            ABCD
            ABCE
            FGHI
            FGHL
            MNOP
            MNOQ
        """
        )
        f.suptitle(session["name"]+f"unit {unit.unit_id} {unit.brain_region}")
        f._save_name = f"unit_{unit.unit_id}_{unit.brain_region}"


        # plot spikes against tracking, speed and angular velocity
        visuals.plot_tracking_xy(downsampled_tracking, ax=axes['A'], plot=True, color=blue_grey, alpha=1)
        axes['A'].scatter(unit_tracking.x, unit_tracking.y, color=unit.color, s=10, zorder=100)
        
        axes['B'].plot(body_tracking.speed, color=blue_grey, lw=2)
        axes['B'].scatter(unit.spikes, unit_tracking.speed, color=colors.speed, s=5, zorder=11)

        axes['C'].plot(body_tracking.angular_velocity, color=blue_grey, lw=2)
        axes['C'].scatter(unit.spikes, unit_tracking.angular_velocity, color=colors.angular_velocity, s=5, zorder=11)

        # plot spikes raster around tone onsets
        visuals.plot_raster(unit.spikes, tone_onsets, axes['D'])

        # plot spike rasters at bouts onsets and offsets
        visuals.plot_raster(unit.spikes, bouts.start_frame, axes['E'])
        visuals.plot_raster(unit.spikes, bouts.end_frame, axes['I'])

        # cleanup and save
        clean_axes(f)
        move_figure(f, 50, 50)
        f.tight_layout()
        axes["A"].set(xlabel="xpos (cm)", ylabel="ypos (cm)", title=f"unit {unit.unit_id} {unit.brain_region}")
        axes['B'].set(xticks=frames, xticklabels=time, xlabel='time (min)', ylabel='speed (cm/s)')
        axes['C'].set(xticks=frames, xticklabels=time, xlabel='time (min)', ylabel='ang vel (deg/s)')
        axes['D'].set(title='tone onset')
        axes['E'].set(title='bout onset')
        axes['I'].set(title='bout offset')

        for ax in 'A':
            axes[ax].axis('equal')

        plt.show()
        break
    break