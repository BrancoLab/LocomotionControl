import sys
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union
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


# window in seconds around events (e.g. for raster plots)


def plot_unit(
    mouse_id: str,
    session_name: str,
    tracking: pd.DataFrame,
    bouts: pd.DataFrame,
    out_bouts: pd.DataFrame,
    in_bouts: pd.DataFrame,
    bouts_stacked: pd.DataFrame,
    units: pd.DataFrame,
    unit_id: Union[int, str],
    tone_onsets: np.ndarray,
    WINDOW: int,
):

    frames = np.arange(0, len(tracking.x), 60 * 60 * 5)
    time = (frames / 60 / 60).astype(np.int32)

    if isinstance(unit_id, int):
        if unit_id not in units.unit_id.values:
            raise ValueError(f"Unit id {unit_id} not in units list:\n{units}")

    for i, unit in units.iterrows():
        if isinstance(unit_id, int):
            if unit.unit_id != unit_id:
                continue
        logger.info(
            f'Showing activity summary for unit unit {i+1}/{len(units)} (id: {unit.unit_id} - in: "{unit.brain_region}")'
        )

        # get tracking data at each spike
        tracking["firing_rate"] = unit.firing_rate
        unit_tracking = data_utils.select_by_indices(tracking, unit.spikes)
        unit_tracking["spikes"] = unit.spikes

        unit_vmax_frate = np.percentile(unit.firing_rate, 98)

        out_bouts_stacked = data_utils.get_bouts_tracking_stacked(
            tracking, out_bouts
        )
        in_bouts_stacked = data_utils.get_bouts_tracking_stacked(
            tracking, in_bouts
        )

        # crate figure
        f = plt.figure(figsize=(24, 12))
        axes = f.subplot_mosaic(
            """
            ARBCDUV
            ARBCEUV
            FSGHIXX
            FSGHLXX
            MTNOPZY
            MTNOQZY
        """
        )
        f.suptitle(session_name + f"unit {unit.unit_id} {unit.brain_region}")
        f._save_name = f"unit_{unit.unit_id}_{unit.brain_region}".replace(
            "\\", "_"
        )

        # plot spikes against tracking, speed and angular velocity
        visuals.plot_heatmap_2d(
            unit_tracking, "spikes", axes["A"], cmap="inferno", vmax=None
        )

        axes["B"].plot(tracking.speed, color=blue_grey, lw=2)
        axes["B"].scatter(
            unit.spikes,
            unit_tracking.speed,
            color=colors.speed,
            s=5,
            zorder=11,
        )

        axes["C"].plot(tracking.dmov_velocity, color=blue_grey, lw=2)
        axes["C"].scatter(
            unit.spikes,
            unit_tracking.dmov_velocity,
            color=colors.dmov_velocity,
            s=5,
            zorder=11,
        )

        # plot spikes heatmap
        visuals.plot_heatmap_2d(
            unit_tracking,
            "firing_rate",
            axes["R"],
            cmap="inferno",
            vmax=unit_vmax_frate,
        )

        # plot spikes raster around tone onsets
        visuals.plot_raster(unit.spikes, tone_onsets, axes["D"], window=WINDOW)
        visuals.plot_aligned(
            tracking.firing_rate,
            tone_onsets,
            axes["E"],
            "aft",
            color=blue_grey,
            lw=1,
            alpha=0.85,
            window=WINDOW,
        )

        # plot spike rasters at bouts onsets and offsets
        visuals.plot_raster(
            unit.spikes, bouts.start_frame, axes["I"], window=WINDOW
        )
        visuals.plot_aligned(
            tracking.firing_rate,
            bouts.start_frame,
            axes["L"],
            "aft",
            color=blue_grey,
            lw=1,
            alpha=0.85,
            window=WINDOW,
        )

        visuals.plot_raster(
            unit.spikes, bouts.end_frame, axes["P"], window=WINDOW
        )
        visuals.plot_aligned(
            tracking.firing_rate,
            bouts.end_frame,
            axes["Q"],
            "pre",
            color=blue_grey,
            lw=1,
            alpha=0.85,
            window=WINDOW,
        )

        # plot firing rate binned by speed and angular velocity
        visuals.plot_bin_x_by_y(
            tracking,
            "firing_rate",
            "speed",
            axes["U"],
            colors=colors.speed,
            bins=10,
            min_count=10,
            s=50,
        )
        visuals.plot_heatmap_2d(
            tracking,
            x_key="speed",
            y_key="firing_rate",
            ax=axes["U"],
            vmax=None,
            zorder=-10,
            alpha=0.5,
            cmap="inferno",
            linewidths=0,
            gridsize=20,
        )

        visuals.plot_bin_x_by_y(
            tracking,
            "firing_rate",
            "dmov_velocity",
            axes["V"],
            colors=colors.dmov_velocity,
            bins=10,
            min_count=10,
            s=50,
        )
        visuals.plot_heatmap_2d(
            tracking,
            x_key="dmov_velocity",
            y_key="firing_rate",
            ax=axes["V"],
            vmax=None,
            zorder=-10,
            alpha=0.5,
            cmap="inferno",
            linewidths=0,
            gridsize=20,
        )
        axes["H"].axvline(0, ls=":", lw=2, color=[0.2, 0.2, 0.2], zorder=101)

        # plot probe electrodes in which there is the unit
        visuals.plot_probe_electrodes(
            db_tables.Unit.get_unit_sites(
                mouse_id,
                session_name,
                unit["unit_id"],
                unit["probe_configuration"],
            ),
            axes["Z"],
            annotate_every=1,
            TARGETS=None,
            x_shift=False,
            s=100,
            lw=2,
        )

        # plot firing rate based on movements
        visuals.plot_avg_firing_rate_based_on_movement(
            tracking, unit, axes["X"]
        )

        # plot heatmap of firing rate vs speed by ang vel heatmap (during bouts)
        trk = dict(
            speed=tracking.speed[tracking.walking == 1],
            dmov_velocity=tracking.dmov_velocity[tracking.walking == 1],
            firing_rate=tracking.firing_rate[tracking.walking == 1],
        )
        visuals.plot_heatmap_2d(
            trk,
            key="firing_rate",
            ax=axes["Y"],
            x_key="speed",
            y_key="dmov_velocity",
            vmax=None,
        )

        # --------------------------------- in bouts --------------------------------- #
        # plot bouts 2d
        visuals.plot_bouts_heatmap_2d(
            tracking, in_bouts, "firing_rate", axes["F"], vmax=unit_vmax_frate
        )

        # plot firing rate binned by global coordinates
        visuals.plot_bin_x_by_y(
            in_bouts_stacked,
            "firing_rate",
            "global_coord",
            axes["S"],
            colors=colors.global_coord,
            bins=10,
            min_count=10,
            s=50,
        )
        visuals.plot_heatmap_2d(
            in_bouts_stacked,
            x_key="global_coord",
            y_key="firing_rate",
            ax=axes["S"],
            vmax=None,
            zorder=-10,
            alpha=0.5,
            cmap="inferno",
            linewidths=0,
            gridsize=20,
        )

        # plot firing rate binned by speed and angular velocity
        visuals.plot_bin_x_by_y(
            in_bouts_stacked,
            "firing_rate",
            "speed",
            axes["G"],
            colors=colors.speed,
            bins=10,
            min_count=10,
            s=50,
        )
        visuals.plot_heatmap_2d(
            in_bouts_stacked,
            x_key="speed",
            y_key="firing_rate",
            ax=axes["G"],
            vmax=None,
            zorder=-10,
            alpha=0.5,
            cmap="inferno",
            linewidths=0,
            gridsize=20,
        )

        visuals.plot_bin_x_by_y(
            in_bouts_stacked,
            "firing_rate",
            "dmov_velocity",
            axes["H"],
            colors=colors.dmov_velocity,
            bins=10,
            min_count=10,
            s=50,
        )
        visuals.plot_heatmap_2d(
            in_bouts_stacked,
            x_key="dmov_velocity",
            y_key="firing_rate",
            ax=axes["H"],
            vmax=None,
            zorder=-10,
            alpha=0.5,
            cmap="inferno",
            linewidths=0,
            gridsize=20,
        )
        axes["H"].axvline(0, ls=":", lw=2, color=[0.2, 0.2, 0.2], zorder=101)

        # --------------------------------- out bouts -------------------------------- #
        visuals.plot_bouts_heatmap_2d(
            tracking, out_bouts, "firing_rate", axes["M"], vmax=unit_vmax_frate
        )

        # plot firing rate binned by global coordinates
        visuals.plot_bin_x_by_y(
            out_bouts_stacked,
            "firing_rate",
            "global_coord",
            axes["T"],
            colors=colors.global_coord,
            bins=10,
            min_count=10,
            s=50,
        )
        visuals.plot_heatmap_2d(
            out_bouts_stacked,
            x_key="global_coord",
            y_key="firing_rate",
            ax=axes["T"],
            vmax=None,
            zorder=-10,
            alpha=0.5,
            cmap="inferno",
            linewidths=0,
            gridsize=20,
        )

        # plot firing rate binned by speed and angular velocity
        visuals.plot_bin_x_by_y(
            out_bouts_stacked,
            "firing_rate",
            "speed",
            axes["N"],
            colors=colors.speed,
            bins=10,
            min_count=10,
            s=50,
        )
        visuals.plot_heatmap_2d(
            out_bouts_stacked,
            x_key="speed",
            y_key="firing_rate",
            ax=axes["N"],
            vmax=None,
            zorder=-10,
            alpha=0.5,
            cmap="inferno",
            linewidths=0,
            gridsize=20,
        )

        visuals.plot_bin_x_by_y(
            out_bouts_stacked,
            "firing_rate",
            "dmov_velocity",
            axes["O"],
            colors=colors.dmov_velocity,
            bins=10,
            min_count=10,
            s=50,
        )
        visuals.plot_heatmap_2d(
            out_bouts_stacked,
            x_key="dmov_velocity",
            y_key="firing_rate",
            ax=axes["O"],
            vmax=None,
            zorder=-10,
            alpha=0.5,
            cmap="inferno",
            linewidths=0,
            gridsize=20,
        )
        axes["H"].axvline(0, ls=":", lw=2, color=[0.2, 0.2, 0.2], zorder=101)

        # ----------------------------- cleanup and save ----------------------------- #
        clean_axes(f)
        set_figure_subplots_aspect(wspace=0.5, hspace=0.6, left=0.3)
        move_figure(f, 50, 50)
        f.tight_layout()
        axes["A"].set(
            xlabel="xpos (cm)",
            ylabel="ypos (cm)",
            title=f"unit {unit.unit_id} {unit.brain_region} | {len(unit.spikes)} spikes",
        )
        axes["B"].set(
            xticks=frames,
            xticklabels=time,
            xlabel="time (min)",
            ylabel="speed (cm/s)",
        )
        axes["C"].set(
            xticks=frames,
            xticklabels=time,
            xlabel="time (min)",
            ylabel="ang vel (deg/s)",
        )
        axes["D"].set(title="tone onset")
        axes["F"].set(title="firing rate", ylabel="IN BOUTS")
        axes["G"].set(ylabel="firing rate", xlabel="speed")
        axes["H"].set(
            ylabel="firing rate", xlabel="angular velocity", xlim=[-350, 350]
        )
        axes["I"].set(title="bout ONSET")
        # axes['L'].set(ylabel='firing rate', xlabel='global coord', xticks=np.arange(0, 1.1, .25))
        axes["M"].set(title="firing rate", ylabel="OUT BOUTS")
        axes["N"].set(ylabel="firing rate", xlabel="speed")
        axes["O"].set(
            ylabel="firing rate", xlabel="angular velocity", xlim=[-350, 350]
        )
        axes["P"].set(title="bouts OFFSET")
        # axes['Q'].set(ylabel='firing rate', xlabel='angular vel')
        axes["R"].set(title="firing rate")
        axes["S"].set(ylabel="firing rate", xlabel="global coord")
        axes["T"].set(ylabel="firing rate", xlabel="global coord")
        axes["U"].set(ylabel="firing rate", xlabel="speed")
        axes["V"].set(
            ylabel="firing rate", xlabel="angular velocity", xlim=[-350, 350]
        )
        axes["Z"].set(xticks=[])

        axes["Y"].set(
            title="bouts firing rate heatmap",
            xlabel="speed (cm/s)",
            ylabel="ang vel (deg/s)",
        )

        for ax in "ARFM":
            axes[ax].axis("equal")
            axes[ax].set(
                xlim=[-5, 45], xticks=[0, 40], ylim=[-5, 65], yticks=[0, 60]
            )
