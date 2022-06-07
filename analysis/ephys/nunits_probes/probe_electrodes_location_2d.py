import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append("./")

from fcutils.plot.figure import clean_axes
from tpd import recorder

from data.dbase.db_tables import Probe
from analysis.visuals import plot_probe_electrodes

base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")
recorder.start(base_folder=base_folder, timestamp=False)

"""
    For each reconstructed probe it plots the position of the electros
"""

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
    "MOs",
    "MOs1",
    "MOs2/3",
    "MOs5",
    "MOs6a",
    "MOs6b",
)

# get data
probes_depths = Probe.fetch("implanted_depth")
probes = pd.DataFrame((Probe * Probe.RecordingSite).fetch())
n_probes = len(probes.mouse_id.unique())

# create figure
f, axes = plt.subplots(
    ncols=n_probes, figsize=(8 * n_probes, 12), sharey=False
)
f._save_name = f"probes"
if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]


# plot each
for probe_n in range(n_probes):
    # select probe sites
    mouse = probes.mouse_id.unique()[probe_n]
    rsites = probes.loc[probes.mouse_id == mouse]

    # plot
    configurations = rsites.probe_configuration.unique()[::-1]
    for config_n, config in enumerate(configurations):
        config_sites = rsites.loc[rsites.probe_configuration == config]
        plot_probe_electrodes(
            config_sites,
            axes[probe_n],
            TARGETS,
            annotate_every=5 if config_n == 0 else False,
            x_pos_delta=config_n / 4,
            x_shift=True if config == "longcol" else False,
        )

        axes[probe_n].text(config_n / 4 + 1, 7800, config)

    axes[probe_n].set(
        title=mouse,
        xlim=[0.5, 1.25 + config_n / 4],
        xticks=np.arange(len(configurations)) / 2,
        xticklabels=configurations,
    )

    # mark probe implanted depth (-175 because of TIP)
    axes[probe_n].axhline(probes_depths[probe_n], lw=4, color="k", alpha=0.5)

clean_axes(f)
recorder.add_figures(svg=False)
plt.show()
