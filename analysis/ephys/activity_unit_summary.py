# imports
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append("./")

from data.dbase.db_tables import Recording, LocomotionBouts

from fcutils.plot.figure import clean_axes

from analysis.ephys.utils import (
    get_data,
    get_clean_walking_onsets,
    get_walking_from_body,
    bin_variable,
)
from analysis.ephys.viz import (
    time_aligned_raster,
    plot_frate_binned_by_var,
    bouts_raster,
)


"""
Makes a summary plot with various views of a single unit's activity.
"""

params = dict(
    MIN_WAKING_DURATION=1.0,  # when the mouse walks < than this we ignore it (seconds)
    MIN_PAUSE_DURATION=1.0,  # when the mouse pauses < before a walking bout than this we ignore it (seconds)
    SPEED_TH=10,  # speed threshold for walking (cm/s)
    min_delta_gcoord=0.5,
    tuning_curves_nbins=20,
)

base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys")


# print all available recordings
recordings = Recording().fetch("name")


for rec in recordings:
    print(f"Processing {rec}")
    # get data
    units, left_fl, right_fl, left_hl, right_hl, body = get_data(rec)
    nunits = len(units)

    # get walking (onset/offset in frames)
    walking = get_walking_from_body(body, params["SPEED_TH"])
    walking_starts, walking_ends = get_clean_walking_onsets(
        walking, params["MIN_WAKING_DURATION"], params["MIN_PAUSE_DURATION"]
    )

    # bin data for tuning curves
    n_bins = params["tuning_curves_nbins"]
    in_bin_speed, bin_values_speed = bin_variable(
        body.speed, bins=np.linspace(-85 / n_bins, 85, n_bins) + 85 / n_bins
    )
    in_bin_avel, bin_values_avel = bin_variable(
        body.thetadot, bins=np.linspace(-500, 500, n_bins) + 500 / n_bins
    )

    # get locomotion bouts
    bouts = LocomotionBouts.get_session_bouts(rec)
    bouts = bouts.loc[
        (bouts.gcoord_delta > params["min_delta_gcoord"])
        & (bouts.direction == "outbound")
    ]
    bouts["gcoord0"] = [body.global_coord[f] for f in bouts.start_frame]
    bouts = bouts.sort_values("gcoord0").reset_index()
    print(f"Kept {len(bouts)} complete locomotion bouts bouts")

    # prepare folders
    rec_fld = base_folder / rec
    rec_fld.mkdir(exist_ok=True)
    rec_svg_fld = rec_fld / "svg"
    rec_svg_fld.mkdir(exist_ok=True)

    for i in range(nunits):
        # get unit
        unit = units.iloc[i]

        # create figure
        fig = plt.figure(figsize=(22, 10))
        axes = fig.subplot_mosaic(
            """
                AABBBB
                CCDDEE
            """
        )

        # plot locomotion onset/offset rasters
        time_aligned_raster(
            axes["A"], unit, walking_starts, t_before=2, t_after=2, dt=0.025
        )
        time_aligned_raster(
            axes["C"], unit, walking_ends, t_before=2, t_after=2, dt=0.025
        )

        # plot tuning curves
        plot_frate_binned_by_var(
            axes["D"],
            unit,
            in_bin_speed,
            bin_values_speed,
            xlabel="Speed (cm/s)",
        )
        plot_frate_binned_by_var(
            axes["E"],
            unit,
            in_bin_avel,
            bin_values_avel,
            color="black",
            xlabel="Angular velocity (deg/s)",
        )

        # plot locomotion bouts raster
        if len(bouts):
            bouts_raster(axes["B"], unit, bouts, body, ds=1)

        # styling
        axes["A"].set(title="Locomotion onset")
        axes["C"].set(title="Locomotion offset")
        axes["B"].set(title="Running bouts")
        axes["D"].set(title="Speed tuning")
        axes["E"].set(title="Angular velocity tuning")

        # save figure
        clean_axes(fig)
        fig.tight_layout()
        region = unit.brain_region.replace("\\", "_")
        if "RSPagl" in region:
            region = "RSP"

        fig.savefig(rec_svg_fld / f"unit_{unit.unit_id}_{region}.svg")
        fig.savefig(rec_fld / f"unit_{unit.unit_id}_{region}.png", dpi=400)
        plt.close(fig)

    #     plt.show()

    #     break

    # break
