# imports
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import medfilt

sys.path.append("./")

from data.dbase.db_tables import Recording, LocomotionBouts

from fcutils.plot.figure import clean_axes

from analysis.ephys.utils import (
    get_data,
    get_clean_walking_onsets,
    get_walking_from_body,
    cleanup_running_bouts,
)
from analysis.ephys.viz import (
    time_aligned_raster,
    # plot_frate_binned_by_var,
    bouts_raster,
    plot_tuning_curves,
)
from analysis.ephys.tuning_curves import (
    get_tuning_curves,
    upsample_farmes_to_ms,
)

"""
Makes a summary plot with various views of a single unit's activity.
"""

params = dict(
    MIN_WAKING_DURATION=1.0,  # when the mouse walks < than this we ignore it (seconds)
    MIN_PAUSE_DURATION=1.0,  # when the mouse pauses < before a walking bout than this we ignore it (seconds)
    SPEED_TH=10,  # speed threshold for walking (cm/s)
    min_delta_gcoord=0.4,
    speed_tuning_curve_bins=np.arange(0, 80, 2),
    avel_tuning_curve_bins=np.arange(-450, 450, 50),
    tuning_curves_repeats=25,
)

base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys")


# print all available recordings
recordings = Recording().fetch("name")


for rec in recordings:
    print(f"Processing {rec}")
    # get data
    try:
        units, left_fl, right_fl, left_hl, right_hl, body = get_data(rec)
    except KeyError:
        print(f"No data for {rec}")
        continue
    nunits = len(units)

    # get walking (onset/offset in frames)
    walking = get_walking_from_body(body, params["SPEED_TH"])
    walking_starts, walking_ends = get_clean_walking_onsets(
        walking, params["MIN_WAKING_DURATION"], params["MIN_PAUSE_DURATION"]
    )

    # smooth data for tuning curves
    speed = medfilt(body.speed, 21)
    avel = medfilt(body.thetadot, 21)

    params["speed_tuning_curve_bins"] = np.arange(
        0, np.max([50, np.percentile(speed, 97.5)]), 2
    )

    # upsample to ms
    speed_ms = upsample_farmes_to_ms(speed)
    avel_ms = upsample_farmes_to_ms(avel)

    # get locomotion bouts
    bouts = LocomotionBouts.get_session_bouts(rec)
    bouts = cleanup_running_bouts(
        bouts, body, min_delta_gcoord=params["min_delta_gcoord"]
    )
    print(f"Kept {len(bouts)} complete locomotion bouts bouts")

    # prepare folders
    rec_fld = base_folder / rec
    rec_fld.mkdir(exist_ok=True)
    rec_svg_fld = rec_fld / "svg"
    rec_svg_fld.mkdir(exist_ok=True)

    for i in range(nunits):
        print(f"        processing unit {i+1}/{nunits}")
        # get unit
        unit = units.iloc[i]

        region = unit.brain_region.replace("\\", "_")
        if "RSP" in region:
            region = "RSP"
        elif "VISp" in region:
            region = "VISp"

        savepath = rec_fld / f"unit_{unit.unit_id}_{region}.png"
        if savepath.exists():
            continue

        # create figure
        fig = plt.figure(figsize=(22, 10))
        axes = fig.subplot_mosaic(
            """
                AABBBB
                AABBBB
                CCDDEE
                CCDDEE
            """
        )

        # plot locomotion onset/offset rasters
        time_aligned_raster(
            axes["A"], unit, walking_starts, t_before=2, t_after=2, dt=0.05
        )
        time_aligned_raster(
            axes["C"], unit, walking_ends, t_before=2, t_after=2, dt=0.05
        )

        # plot tuning curves
        speed_tuning_curves = get_tuning_curves(
            unit.spikes_ms, speed_ms, params["speed_tuning_curve_bins"]
        )
        plot_tuning_curves(
            axes["D"], speed_tuning_curves, unit.color, xlabel="speed (cm/s)"
        )

        avel_tuning_curves = get_tuning_curves(
            unit.spikes_ms, avel_ms, params["avel_tuning_curve_bins"]
        )
        plot_tuning_curves(
            axes["E"], avel_tuning_curves, "black", xlabel="avel (deg/s)"
        )

        # plot locomotion bouts raster
        if len(bouts):
            bouts_raster(axes["B"], unit, bouts, body, ds=2)

        # styling
        axes["A"].set(title="Locomotion onset")
        axes["C"].set(title="Locomotion offset")
        axes["B"].set(title="Running bouts")
        axes["D"].set(title="Speed tuning", xlim=[0, 80])
        axes["E"].set(title="Angular velocity tuning")

        # save figure
        clean_axes(fig)
        fig.tight_layout()

        # fig.savefig(rec_svg_fld / f"unit_{unit.unit_id}_{region}.svg")
        fig.savefig(savepath, dpi=400)
        plt.close(fig)

    #     plt.show()

    #     break

    # break