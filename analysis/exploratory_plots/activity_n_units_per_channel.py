import sys
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import numpy as np

sys.path.append("./")

from fcutils.plot.figure import clean_axes
from myterial import blue_grey

from analysis.visuals import plot_probe_electrodes

base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")


# ---------------------------------------------------------------------------- #
#                                  plot probes                                 #
# ---------------------------------------------------------------------------- #
# get all recordings


def plot_n_units_per_channel(rname, units, rsites, TARGETS):
    logger.info(f'Plotting n units per channel for {rname}')
    f, axes = plt.subplots(figsize=(12, 12), ncols=2, sharey=True)
    f.suptitle(rname)
    f._save_name = f"activity_units_per_channel"

    # draw probe
    plot_probe_electrodes(rsites, axes[0], TARGETS)

    # draw barplot of # units per channel
    counts = units.groupby("site_id").count()["name"]
    _colors = [
        rsites.loc[rsites.site_id == n]["color"].iloc[0] for n in counts.index
    ]
    _regions = [
        rsites.loc[rsites.site_id == n]["brain_region"].iloc[0]
        for n in counts.index
    ]
    colors = [
        c if r in TARGETS else ('k' if r in ('unknown', 'OUT') else blue_grey)
        for c, r in zip(_colors, _regions)
    ]
    probe_coords = [
        rsites.loc[rsites.site_id == n]["probe_coordinates"].iloc[0]
        for n in counts.index
    ]


    axes[1].scatter(
        counts.values + np.random.normal(0, .02, size=len(counts.values)), 
        probe_coords, 
        color=colors, s=100, lw=1, ec="k"
    )

    for x, y in zip(counts.values, probe_coords):
        axes[1].plot(
            [0, x], 
            [y, y], 
            color=[.2, .2, .2], lw=2, zorder=-1
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



if __name__ == "__main__":
    import sys
    sys.path.append('./')
    from data.dbase import db_tables
    import pandas as pd


    # ----------------------------- load probe/units ----------------------------- #
    session = 'FC_210721_AAA1110750_hairpin'
    recording = (db_tables.Recording & f'name="{session}"').fetch(as_dict=True)[0]
    cf = recording['recording_probe_configuration']
    logger.info("Fetching ephys data")
    units = db_tables.Unit.get_session_units(
        session,
        cf,
        spikes=True,
        firing_rate=True,
        frate_window=100,
    )
    units['probe_configuration'] = [cf] * len(units)
    rsites = pd.DataFrame((db_tables.Probe.RecordingSite & recording & f'probe_configuration="{cf}"').fetch())
    logger.info(f'Found {len(units)} units')


    plot_n_units_per_channel(recording, units, rsites, ['CUN'])
    
    plt.show()