import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.append("./")

from fcutils.plot.figure import clean_axes
from tpd import recorder
from myterial import blue_grey

from data.dbase.db_tables import Probe, Unit, Recording, Tracking


base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")

TARGETS = (
    "PRNr",
    "PRNc",
    "CUN",
    "GRN",
    "MB",
    "PPN",
    "RSPagl2/3",
    "RSPagl5",
    "RSPagl6",
    "RSPd1",
    "RSPd2",
)

# ---------------------------------------------------------------------------- #
#                                  plot probes                                 #
# ---------------------------------------------------------------------------- #
# get all recordings
recordings = pd.DataFrame(Recording.fetch())

for i, recording in recordings.iterrows():
    # start logging
    rname = recording["name"]
    recorder.start(base_folder=base_folder, name=rname, timestamp=False)

    # get units
    units = pd.DataFrame((Unit * Unit.Spikes & dict(recording)).fetch())
    logger.info(f'processing recording: "{rname}" - {len(units)} units')

    # get tracking data
    # load tracking data
    tracking = Tracking.get_session_tracking(rname, body_only=True)

    # get probe
    rsites = pd.DataFrame((Probe.RecordingSite & dict(recording)).fetch())

    for n, unit in units.iterrows():
        x, y, speed = (
            tracking["x"].values,
            tracking["y"].values,
            tracking["speed"].values,
        )
        avel, orientation = (
            tracking["angular_velocity"].values,
            tracking["orientation"].values,
        )
        time = np.arange(len(x))
        spike_times = unit.spikes
        spike_times = spike_times[
            spike_times < len(x)
        ]  # ! remove and figure out why necessary
        color = rsites.iloc[unit["site_id"]].color

        # make figure
        f = plt.figure(figsize=(18, 12))
        f.suptitle(
            f'unit {unit["unit_id"]} - {rsites.iloc[unit["site_id"]].brain_region}'
        )
        axes_dict = f.subplot_mosaic(
            """
            ABC
            ABD
            """
        )

        # plot against XY tracking
        axes_dict["A"].plot(x, y, color=blue_grey)
        axes_dict["A"].scatter(
            x[spike_times], y[spike_times], color=color, zorder=100, alpha=0.5
        )

        # plot against speed
        axes_dict["C"].plot(time, speed, color=blue_grey)
        axes_dict["C"].scatter(
            time[spike_times],
            speed[spike_times],
            color=color,
            zorder=100,
            alpha=0.5,
        )

        # plot against speed
        axes_dict["D"].plot(time, avel, color=blue_grey)
        axes_dict["D"].scatter(
            time[spike_times],
            avel[spike_times],
            color=color,
            zorder=100,
            alpha=0.5,
        )

        # cleanup and save
        axes_dict["A"].set(xlabel="xpos (cm)", ylabel="ypos (cm)")
        axes_dict["C"].set(xlabel="time (frames)", ylabel="speed (cm/s)")
        clean_axes(f)

        plt.show()
        # break
