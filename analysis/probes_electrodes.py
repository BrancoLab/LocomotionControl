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
recorder.start(base_folder=base_folder, name='probes_reconstructions', timestamp=False)


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

# get data
probes = pd.DataFrame((Probe * Probe.RecordingSite).fetch())
n_probes = len(probes.mouse_id.unique())

# create figure
f, axes = plt.subplots(ncols=n_probes, figsize=(4 * n_probes, 12), sharey=True)
f._save_name = f"probes"
if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]

# plot each
for probe_n in range(n_probes):
    # select probe sites
    mouse = probes.mouse_id.unique()[probe_n]
    rsites = probes.loc[probes.mouse_id == mouse]

    # plot
    plot_probe_electrodes(rsites, axes[probe_n], TARGETS)
    axes[probe_n].set(title=mouse)

clean_axes(f)
# plt.show()
recorder.add_figures(svg=False)