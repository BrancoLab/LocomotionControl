import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.append("./")

from fcutils.plot.figure import clean_axes
from tpd import recorder
from myterial import blue_grey

from data.dbase.db_tables import Probe, Unit, Recording
from analysis.visuals import plot_probe_electrodes

base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")

TARGETS = (
    "PRNr",
    "PRNc",
    "CUN",
    "GRN",
    "MB",
    "PPN",
    "RSPagl1",
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
    f, axes = plt.subplots(figsize=(12, 12), ncols=2, sharey=True)
    f.suptitle(rname)
    f._save_name = f"{rname}_activity_preview"

    # draw probe
    plot_probe_electrodes(rsites, axes[0], TARGETS)

    # draw barplot of # units per channel
    counts = units.groupby("site_id").count()["name"]
    _colors = [rsites.loc[rsites.site_id == n]['color'].iloc[0] for n in counts.index]
    _regions = [rsites.loc[rsites.site_id == n]['brain_region'].iloc[0] for n in counts.index]
    colors = [
        c if r in TARGETS else (c if c == "k" else blue_grey)
        for c, r in zip(_colors, _regions)
    ]
    probe_coords = [rsites.loc[rsites.site_id == n]['probe_coordinates'].iloc[0] for n in counts.index]
    axes[1].barh(
        probe_coords,
        counts.values,
        color=colors,
        height=10,
        lw=1,
        ec='k'
    )

    # cleanup and save
    axes[0].set(
        ylabel="Probe position (um)",
        xticks=[],
        xlim=[0.5, 1.5],
        ylim=[0, 8000],
    )
    axes[1].set(xlabel="# units per channel", ylim=[0, 8000])

    clean_axes(f)
    plt.show()

    recorder.add_figures(svg=False)
    plt.close("all")
