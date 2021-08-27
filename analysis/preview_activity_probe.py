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

from data.dbase.db_tables import Probe, Unit, Recording


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

    # get probe
    rsites = pd.DataFrame((Probe.RecordingSite & dict(recording)).fetch())

    # -------------------------------- draw probe -------------------------------- #
    f = plt.figure(figsize=(12, 12))
    f.suptitle(rname)
    f._save_name = f"{rname}_activity_preview"

    axes_dict = f.subplot_mosaic(
        """
            AB
            AB
        """
    )

    # draw probe
    x = np.ones(len(rsites))
    colors = [
        rs.color
        if rs.brain_region in TARGETS
        else ("k" if rs.color == "k" else blue_grey)
        for i, rs in rsites.iterrows()
    ]
    axes_dict["A"].scatter(
        x,
        rsites.probe_coordinates,
        s=50,
        lw=0.5,
        ec=[0.3, 0.3, 0.3],
        marker="s",
        c=colors,
    )

    for i in range(len(x)):
        if i % 5 == 0:
            axes_dict["A"].annotate(
                f"{i} - {rsites.brain_region.iloc[i]}",
                (x[i] - 0.3, rsites.probe_coordinates.iloc[i]),
            )

    # draw barplot of # units per channel
    counts = units.groupby("site_id").count()["name"]
    _colors = rsites.color.iloc[counts.index].values
    _regions = rsites.brain_region.iloc[counts.index].values
    colors = [
        c if r in TARGETS else (c if c == "k" else blue_grey)
        for c, r in zip(_colors, _regions)
    ]
    axes_dict["B"].barh(
        rsites.probe_coordinates.iloc[counts.index].values,
        counts.values,
        color=colors,
        height=40,
    )

    # cleanup and save
    axes_dict["A"].set(
        ylabel="Probe position (um)",
        xticks=[],
        xlim=[0.5, 1.5],
        ylim=[0, 8000],
    )
    axes_dict["B"].set(xlabel="# units per channel", ylim=[0, 8000])

    clean_axes(f)
    # plt.show()

    recorder.add_figures(svg=False)
    plt.close("all")
