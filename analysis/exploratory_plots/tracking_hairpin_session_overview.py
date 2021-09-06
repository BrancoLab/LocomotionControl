import sys
from tpd import recorder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append("./")
from pathlib import Path

from fcutils.plot.figure import clean_axes
from myterial.utils import make_palette
from myterial import grey_darker, grey_light,amber_light, amber_darker, grey_dark

from analysis import visuals
from analysis._visuals import move_figure

from data.dbase import db_tables
from data import data_utils, colors

base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")

"""
    Plot a few exploratory plots for each session a recording was performed on.

        1. Overview of tracking data  
"""


def plot_hairpin_tracking(
            session_name:str,
            tracking:pd.DataFrame,
            downsampled_tracking:pd.DataFrame,
            bouts:pd.DataFrame,
            out_bouts:pd.DataFrame,
            in_bouts:pd.DataFrame,
            out_bouts_stacked:pd.DataFrame,
            in_bouts_stacked:pd.DataFrame,
    ):

    
    # crate figure
    f = plt.figure(figsize=(24, 12))
    axes = f.subplot_mosaic(
        """
        ABCDER
        ABCDER
        FGPHHS
        FGPIIT
        LMQNNU
        LMQOOV
    """
    )
    f.suptitle(session_name)
    f._save_name = "tracking_data_2d"

    # draw tracking 2D
    visuals.plot_tracking_xy(downsampled_tracking, ax=axes['A'], plot=True, color=[.6, .6, .6], alpha=.5)
    visuals.plot_bouts_2d(tracking, bouts, axes['A'], lw=2, zorder=100)

    # draw tracking 1D
    visuals.plot_tracking_linearized(downsampled_tracking, ax=axes['B'], plot=True, color=[.6, .6, .6])
    visuals.plot_bouts_1d(tracking, bouts, axes['B'], lw=3, zorder=100)

    # plot speed aligned to bouts starts and ends
    visuals.plot_aligned(tracking.speed, bouts.start_frame, axes['C'], 'after', alpha=.5)
    visuals.plot_aligned(tracking.speed, bouts.end_frame, axes['D'], 'pre', alpha=.5)

    # plot histograms of bouts durations
    axes['E'].hist(out_bouts.duration, color=colors.outbound, label='out', bins=15, alpha=.7, ec=[.2, .2, .2], histtype='stepfilled', lw=2)
    axes['E'].hist(in_bouts.duration, color=colors.inbound, label='in', bins=15, alpha=.7, ec=[.2, .2, .2], histtype='stepfilled', lw=2)

    # plot speed vs angular velocity
    is_locomoting = np.where(db_tables.LocomotionBouts.is_locomoting(session_name))[0]
    trk = pd.DataFrame(dict(speed=tracking.speed[is_locomoting], angular_velocity=np.abs(tracking.angular_velocity[is_locomoting])))
    visuals.plot_bin_x_by_y(trk, 'angular_velocity', 'speed', axes['R'], bins=np.linspace(0, trk.speed.max(), 11), colors=grey_darker)

    # draw speed and orientation heatmaps during bouts
    visuals.plot_heatmap_2d(in_bouts_stacked, 'speed', ax=axes['F'], alpha=1, vmax=30, cmap='inferno')
    visuals.plot_heatmap_2d(out_bouts_stacked, 'speed', ax=axes['L'], alpha=1, vmax=30, cmap='inferno')
    visuals.plot_heatmap_2d(in_bouts_stacked, 'orientation', ax=axes['G'], alpha=1, vmin=0, vmax=360, edgecolors=grey_darker)
    visuals.plot_heatmap_2d(out_bouts_stacked, 'orientation', ax=axes['M'], alpha=1, vmin=0, vmax=360, edgecolors=grey_darker)
    visuals.plot_heatmap_2d(in_bouts_stacked, 'angular_velocity', ax=axes['P'], alpha=1, vmin=-45, vmax=45, edgecolors=grey_darker)
    visuals.plot_heatmap_2d(out_bouts_stacked, 'angular_velocity', ax=axes['Q'], alpha=1, vmin=-45, vmax=45, edgecolors=grey_darker)


    # plot speeds binned by global coords for in/out bouts
    nbins=25
    clrs=make_palette(grey_light, grey_dark, nbins-1)
    clrs2=make_palette(amber_light, amber_darker, nbins-1)

    visuals.plot_bin_x_by_y(in_bouts_stacked, 'speed', 'global_coord', axes['H'], bins=np.linspace(0, 1, nbins), colors=clrs)
    visuals.plot_bin_x_by_y(in_bouts_stacked, 'angular_velocity', 'global_coord', axes['I'], bins=np.linspace(0, 1, nbins), colors=clrs2)
    visuals.plot_bin_x_by_y(out_bouts_stacked, 'speed', 'global_coord', axes['N'], bins=np.linspace(0, 1, nbins), colors=clrs)
    visuals.plot_bin_x_by_y(out_bouts_stacked, 'angular_velocity', 'global_coord', axes['O'], bins=np.linspace(0, 1, nbins), colors=clrs2)

    # plot histograms with speed and angular velocity
    axes['S'].hist(in_bouts_stacked.speed, bins=20, color=colors.speed)
    axes['T'].hist(in_bouts_stacked.angular_velocity, bins=20, color=colors.angular_velocity)
    axes['U'].hist(out_bouts_stacked.speed, bins=20, color=colors.speed)
    axes['V'].hist(out_bouts_stacked.angular_velocity, bins=20, color=colors.angular_velocity)

    for ax in 'IO':
        axes[ax].axhline(0, lw=1, color=[.6, .6, .6], zorder=-1)

    # plot bouts centered
    # visuals.plot_bouts_x_by_y(tracking, in_bouts, axes['S'], 'speed', 'global_coord')

    # cleanup and save
    clean_axes(f)
    move_figure(f, 50, 50)
    f.tight_layout()

    axes["A"].set(xlabel="xpos (cm)", ylabel="ypos (cm)", title=f'{len(in_bouts)} IN - {len(out_bouts)} OUT')
    axes["B"].set(ylabel="time in exp", xlabel="arena position")
    axes["C"].set(ylabel='speed (cm/s)', xticks=[0, 60, 120], xticklabels=[-1, 0, 1], xlabel='time (s)', title='bout onset')
    axes["D"].set(ylabel='speed (cm/s)', xticks=[0, 60, 120], xticklabels=[-1, 0, 1], xlabel='time (s)', title='bout offset')
    axes['E'].set(ylabel='counts', xlabel='duration (s)', title='Bouts duration')
    axes['F'].set(ylabel='IN', xticks=[], yticks=[])
    axes['G'].set( xticks=[], yticks=[])
    axes['H'].set(ylabel='speed (cm/s)', xticks=[])
    axes['I'].set(ylabel='ang vel (deg/s)', xticks=[])
    axes['L'].set(ylabel='OUT', xticks=[], yticks=[], xlabel='speed')
    axes['M'].set( xticks=[], yticks=[], xlabel='orientation')
    axes['N'].set(ylabel='speed (cm/s)', xticks=[])
    axes['O'].set(ylabel='ang vel (deg/s)', xticks=np.linspace(0, 1, 11))
    axes['P'].set( xticks=[], yticks=[])
    axes['Q'].set( xticks=[], yticks=[], xlabel='ang vel (deg/s)')
    axes['R'].set(xlabel='speed (cm/s)', ylabel='abs(ang vel) (deg/s)')

    axes['S'].set(xlabel='speed (cm/s)')
    axes['T'].set(xlabel='angular velocity (deg/s)')
    axes['U'].set(xlabel='speed (cm/s)')
    axes['V'].set(xlabel='angular velocity (deg/s)')

    for ax in "AFGLMPQ":
        axes[ax].axis('equal')
        axes[ax].set(xlim=[-5, 45], ylim=[-5, 65], xticks=[0, 40], yticks=[0, 60])