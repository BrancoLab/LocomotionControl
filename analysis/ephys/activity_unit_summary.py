
# imports
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt

sys.path.append("./")

from data.dbase.db_tables import Probe, Unit, Session, ValidatedSession, Recording, Tracking

from fcutils.maths.signals import get_onset_offset
from fcutils.plot.figure import clean_axes, calc_nrows_ncols
from myterial import amber_darker, green_dark, grey_darker, grey_dark

from analysis.ephys.utils import get_data, get_clean_walking_onsets, get_walking_from_body, bin_variable
from analysis.ephys.viz import time_aligned_raster, plot_frate_binned_by_var


"""
Makes a summary plot with various views of a single unit's activity.
"""

params = dict(
  MIN_WAKING_DURATION = 1.0,    # when the mouse walks < than this we ignore it (seconds)
    MIN_PAUSE_DURATION = .5,    # when the mouse pauses < before a walking bout than this we ignore it (seconds)
    SPEED_TH = 10,              # speed threshold for walking (cm/s)  
)

base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys")


# print all available recordings
recordings = Recording().fetch("name")


for rec in recordings:
    # get data
    units, left_fl, right_fl, left_hl, right_hl, body = get_data(rec)
    nunits = len(units)

    # get walking (onset/offset in frames)
    walking = get_walking_from_body(body, params["SPEED_TH"])
    walking_starts, walking_ends = get_clean_walking_onsets(walking, params["MIN_WAKING_DURATION"], params["MIN_PAUSE_DURATION"])

    # TODO test if walking starts/ends timestamps are in frames or seconds

    # bin data for tuning curves
    n_bins = 20
    in_bin_speed, bin_values_speed = bin_variable(body.speed, bins=np.linspace(-85/n_bins, 85, n_bins)+85/n_bins)
    in_bin_avel, bin_values_avel = bin_variable(body.thetadot, bins=np.linspace(-500, 500, n_bins)+500/n_bins)

    # TODO get locomotion bouts


    # prepare folders
    rec_fld = base_folder/rec
    rec_fld.mkdir(exist_ok=True)
    rec_svg_fld = rec_fld/"svg"
    rec_svg_fld.mkdir(exist_ok=True)

    for i in range(nunits):
        # get unit
        unit = units.iloc[i]
        unit_savepath = rec_fld/f"unit_{unit.id}_{unit.brain_area}"

        # create figure
        fig = plt.figure(figsize=(16, 9))
        axes = fig.subplot_mosaic(
            """
                AABBBB
                CCDDEE
            """
        )

        # plot locomotion onset/offset rasters
        time_aligned_raster(
            axes["A"], unit, walking_starts, t_before=2, t_after=2, dt=.025
        )
        time_aligned_raster(
            axes["B"], unit, walking_ends, t_before=2, t_after=2, dt=.025
        )

        # plot tuning curves
        plot_frate_binned_by_var(
            axes["D"], unit, in_bin_speed, bin_values_speed, xlabel="Speed (cm/s)"
        )
        plot_frate_binned_by_var(
            axes["E"], unit, in_bin_avel, bin_values_avel, xlabel="Angular velocity (deg/s)"
        )


        # styling
        axes["A"].set(title="Locomotion onset")
        axes["B"].set(title="Locomotion offset")
        axes["C"].set(title="Speed tuning")
        axes["D"].set(title="Angular velocity tuning")

        # save figure
        fig.savefig(rec_svg_fld/f"unit_{unit.id}_{unit.brain_area}.svg")
        fig.savcefig(rec_fld/f"unit_{unit.id}_{unit.brain_area}.png")
        plt.close(fig)

        plt.show()

        break