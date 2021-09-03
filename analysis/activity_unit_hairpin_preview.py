import sys
from tpd import recorder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from loguru import logger

sys.path.append("./")
from pathlib import Path

from fcutils.plot.figure import clean_axes, set_figure_subplots_aspect
from myterial import blue_grey

from analysis import visuals
from analysis._visuals import move_figure

from data.dbase import db_tables
from data import data_utils, colors

base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")




WINDOW: int = int(3 * 60)  # window in seconds around events (e.g. for raster plots)


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
    units = db_tables.Unit.get_session_units(session['name'], spikes=True, firing_rate=True, frate_window=500)

    # get tone onset times
    tone_onsets = db_tables.Behavior.get_session_onsets(session['name'], 'speaker')
    logger.debug(f'Found {len(tone_onsets)} tones in session')

    for i, unit in units.iterrows():
        logger.info(f'Showing unit {i+1}/{len(units)}')

        # get tracking data at each spike
        body_tracking['firing_rate'] = unit.firing_rate
        unit_tracking = data_utils.select_by_indices(body_tracking, unit.spikes)
        unit_tracking['spikes'] = unit.spikes

        unit_vmax_frate = np.percentile(unit.firing_rate, 98)
        logger.info(f'Firing rate 98th percentile: {unit_vmax_frate:.3f}')

        out_bouts_stacked = data_utils.get_bouts_tracking_stacked(body_tracking, out_bouts)
        in_bouts_stacked = data_utils.get_bouts_tracking_stacked(body_tracking, in_bouts)

        # crate figure
        f = plt.figure(figsize=(24, 12))
        axes = f.subplot_mosaic(
            """
            ARBCDU
            ARBCEU
            FSGHIV
            FSGHLV
            MTNOPZ
            MTNOQZ
        """
        )
        f.suptitle(session["name"]+f"unit {unit.unit_id} {unit.brain_region}")
        f._save_name = f"unit_{unit.unit_id}_{unit.brain_region}"


        # plot spikes against tracking, speed and angular velocity
        visuals.plot_heatmap_2d(unit_tracking, 'spikes', axes['A'], cmap='inferno', vmax=None)
        
        axes['B'].plot(body_tracking.speed, color=blue_grey, lw=2)
        axes['B'].scatter(unit.spikes, unit_tracking.speed, color=colors.speed, s=5, zorder=11)

        axes['C'].plot(body_tracking.angular_velocity, color=blue_grey, lw=2)
        axes['C'].scatter(unit.spikes, unit_tracking.angular_velocity, color=colors.angular_velocity, s=5, zorder=11)

        # plot spikes heatmap
        visuals.plot_heatmap_2d(unit_tracking, 'firing_rate', axes['R'], cmap='inferno', vmax=unit_vmax_frate)

        # plot spikes raster around tone onsets
        visuals.plot_raster(unit.spikes, tone_onsets, axes['D'], window=WINDOW)
        visuals.plot_aligned(body_tracking.firing_rate, tone_onsets, axes['E'], 'aft', color = blue_grey, lw=1, alpha=.85, window=WINDOW)

        # plot spike rasters at bouts onsets and offsets
        visuals.plot_raster(unit.spikes, bouts.start_frame, axes['I'], window=WINDOW)
        visuals.plot_aligned(body_tracking.firing_rate, bouts.start_frame, axes['L'], 'aft', color = blue_grey, lw=1, alpha=.85, window=WINDOW)

        visuals.plot_raster(unit.spikes, bouts.end_frame, axes['P'], window=WINDOW)
        visuals.plot_aligned(body_tracking.firing_rate, bouts.end_frame, axes['Q'], 'pre', color = blue_grey, lw=1, alpha=.85, window=WINDOW)

        # plot firing rate by speed and angular velocity
        # plot firing rate binned by speed and angular velocity
        visuals.plot_bin_x_by_y(body_tracking, 'firing_rate', 'speed', axes['U'], colors=colors.speed, bins=10, min_count=10, s=50)
        visuals.plot_heatmap_2d(body_tracking, x_key='speed', y_key='firing_rate', ax=axes['U'], vmax=None, zorder=-10, alpha=.5, cmap='inferno', linewidths=0, gridsize=20)

        visuals.plot_bin_x_by_y(body_tracking, 'firing_rate', 'angular_velocity', axes['V'], colors=colors.angular_velocity, bins=10, min_count=10, s=50)
        visuals.plot_heatmap_2d(body_tracking, x_key='angular_velocity', y_key='firing_rate', ax=axes['V'], vmax=None, zorder=-10, alpha=.5, cmap='inferno', linewidths=0, gridsize=20)
        axes['H'].axvline(0, ls=':', lw=2, color=[.2, .2, .2], zorder=101)

        # plot probe electrodes in which there is the unit
        visuals.plot_probe_electrodes(db_tables.Unit.get_unit_sites(session['mouse_id'], session["name"], unit['unit_id']), axes['Z'], annotate_every=1, TARGETS=None, x_shift=False, s=100, lw=2)

        # --------------------------------- in bouts --------------------------------- #
        # plot bouts 2d
        visuals.plot_bouts_heatmap_2d(body_tracking, in_bouts, 'firing_rate', axes['F'], vmax=unit_vmax_frate)

        # plot firing rate binned by global coordinates
        visuals.plot_bin_x_by_y(in_bouts_stacked, 'firing_rate', 'global_coord', axes['S'], colors=colors.global_coord, bins=10, min_count=10, s=50)
        visuals.plot_heatmap_2d(in_bouts_stacked, x_key='global_coord', y_key='firing_rate', ax=axes['S'], vmax=None, zorder=-10, alpha=.5, cmap='inferno', linewidths=0, gridsize=20)

        # plot firing rate binned by speed and angular velocity
        visuals.plot_bin_x_by_y(in_bouts_stacked, 'firing_rate', 'speed', axes['G'], colors=colors.speed, bins=10, min_count=10, s=50)
        visuals.plot_heatmap_2d(in_bouts_stacked, x_key='speed', y_key='firing_rate', ax=axes['G'], vmax=None, zorder=-10, alpha=.5, cmap='inferno', linewidths=0, gridsize=20)

        visuals.plot_bin_x_by_y(in_bouts_stacked, 'firing_rate', 'angular_velocity', axes['H'], colors=colors.angular_velocity, bins=10, min_count=10, s=50)
        visuals.plot_heatmap_2d(in_bouts_stacked, x_key='angular_velocity', y_key='firing_rate', ax=axes['H'], vmax=None, zorder=-10, alpha=.5, cmap='inferno', linewidths=0, gridsize=20)
        axes['H'].axvline(0, ls=':', lw=2, color=[.2, .2, .2], zorder=101)

        # --------------------------------- out bouts -------------------------------- #
        visuals.plot_bouts_heatmap_2d(body_tracking, out_bouts, 'firing_rate', axes['M'],vmax=unit_vmax_frate)

        # plot firing rate binned by global coordinates
        visuals.plot_bin_x_by_y(out_bouts_stacked, 'firing_rate', 'global_coord', axes['T'], colors=colors.global_coord, bins=10, min_count=10, s=50)
        visuals.plot_heatmap_2d(out_bouts_stacked, x_key='global_coord', y_key='firing_rate', ax=axes['T'], vmax=None, zorder=-10, alpha=.5, cmap='inferno', linewidths=0, gridsize=20)

        # plot firing rate binned by speed and angular velocity
        visuals.plot_bin_x_by_y(out_bouts_stacked, 'firing_rate', 'speed', axes['N'], colors=colors.speed, bins=10, min_count=10, s=50)
        visuals.plot_heatmap_2d(out_bouts_stacked, x_key='speed', y_key='firing_rate', ax=axes['N'], vmax=None, zorder=-10, alpha=.5, cmap='inferno', linewidths=0, gridsize=20)

        visuals.plot_bin_x_by_y(out_bouts_stacked, 'firing_rate', 'angular_velocity', axes['O'], colors=colors.angular_velocity, bins=10, min_count=10, s=50)
        visuals.plot_heatmap_2d(out_bouts_stacked, x_key='angular_velocity', y_key='firing_rate', ax=axes['O'], vmax=None, zorder=-10, alpha=.5, cmap='inferno', linewidths=0, gridsize=20)
        axes['H'].axvline(0, ls=':', lw=2, color=[.2, .2, .2], zorder=101)

        # TODO add visualization of speeds distributions to tracking figure

        # ----------------------------- cleanup and save ----------------------------- #
        clean_axes(f)
        set_figure_subplots_aspect(wspace=.5, hspace=.6, left=.3)
        move_figure(f, 50, 50)
        f.tight_layout()
        axes["A"].set(xlabel="xpos (cm)", ylabel="ypos (cm)", title=f"unit {unit.unit_id} {unit.brain_region} | {len(unit.spikes)} spikes")
        axes['B'].set(xticks=frames, xticklabels=time, xlabel='time (min)', ylabel='speed (cm/s)')
        axes['C'].set(xticks=frames, xticklabels=time, xlabel='time (min)', ylabel='ang vel (deg/s)')
        axes['D'].set(title='tone onset')
        axes['F'].set(title='firing rate', ylabel='IN BOUTS')
        axes['G'].set(ylabel='firing rate', xlabel='speed')
        axes['H'].set(ylabel='firing rate', xlabel='angular velocity', xlim=[-350, 350])
        axes['I'].set(title='bout ONSET')
        # axes['L'].set(ylabel='firing rate', xlabel='global coord', xticks=np.arange(0, 1.1, .25))
        axes['M'].set(title='firing rate', ylabel='OUT BOUTS')
        axes['N'].set(ylabel='firing rate', xlabel='speed')
        axes['O'].set(ylabel='firing rate', xlabel='angular velocity', xlim=[-350, 350])
        axes['P'].set(title='bouts OFFSET')
        # axes['Q'].set(ylabel='firing rate', xlabel='angular vel')
        axes['R'].set(title='firing rate')
        axes['S'].set(ylabel='firing rate', xlabel='global coord')
        axes['T'].set(ylabel='firing rate', xlabel='global coord')
        axes['U'].set(ylabel='firing rate', xlabel='speed')
        axes['V'].set(ylabel='firing rate', xlabel='angular velocity', xlim=[-350, 350])
        axes['Z'].set(xticks=[])

        for ax in 'ARFM':
            axes[ax].axis('equal')
            axes[ax].set(xlim=[-5, 45], xticks=[0, 40], ylim=[-5, 65], yticks=[0, 60])

        plt.show()
        # break

        # recorder.add_figures(svg=False)
        # plt.close("all")

    break