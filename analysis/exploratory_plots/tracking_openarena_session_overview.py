import sys
from tpd import recorder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append("./")
from pathlib import Path

from fcutils.plot.figure import clean_axes
from myterial.utils import make_palette
from myterial import grey_dark, grey_light

from analysis import visuals

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
    & 'arena="openarena"'
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
    downsampled_tracking = db_tables.Tracking.get_session_tracking(
        session["name"], body_only=True
    )
    data_utils.downsample_tracking_data(downsampled_tracking, factor=10)

    # get locomotion bouts
    bouts = db_tables.LocomotionBouts.get_session_bouts(session['name'])

    # crate figure
    f = plt.figure(figsize=(24, 12))
    axes = f.subplot_mosaic(
        """
        ABCD
        EFGH
    """
    )
    f.suptitle(session["name"])
    f._save_name = "tracking_data_2d"


    # plot tracking and botus 2d
    visuals.plot_tracking_xy(downsampled_tracking, ax=axes['A'], plot=True, color=[.4, .4, .4], alpha=.8)
    visuals.plot_bouts_2d(body_tracking, bouts, axes['A'], lw=2, zorder=100, c='salmon')

    # plot speed aligned to bouts starts and ends
    visuals.plot_aligned(body_tracking.speed, bouts.start_frame, axes['B'], 'after', alpha=.5)
    visuals.plot_aligned(body_tracking.speed, bouts.end_frame, axes['C'], 'pre', alpha=.5)

    # plot histopgram of botus duration
    axes['D'].hist(bouts.duration, color=grey_dark, label='out', bins=15, alpha=.7, ec=[.2, .2, .2], histtype='stepfilled', lw=2)

    # TODO plot bouts centered and rotated in E

    # plot bouts speed and ang vel profiles
    visuals.plot_bouts_x(body_tracking, bouts, axes['F'], 'speed', color=colors.speed, alpha=.5)
    visuals.plot_bouts_x(body_tracking, bouts, axes['G'], 'angular_velocity', color=colors.angular_velocity, alpha=.5)


    plt.show()
    break
    recorder.add_figures(svg=False)
    plt.close("all")
