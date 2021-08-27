import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append("./")

# from fcutils.plot.figure import clean_axes

from data.dbase.db_tables import Probe


base_folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")

# ---------------------------------------------------------------------------- #
#                                  plot probes                                 #
# ---------------------------------------------------------------------------- #
# get probes data
probes = Probe().fetch(as_dict=True)

for probe in probes:
    # get recording sites
    rsites = pd.DataFrame((Probe.RecordingSite & probe).fetch())

    x = np.ones(len(rsites))
    x[::3] = 1.2
    x[::2] = x[::2] - .4

    f, ax = plt.subplots(figsize=(8, 12))
    ax.scatter(
        x,
        rsites.probe_coordinates,
        s=50,
        lw=0.5,
        ec=[0.3, 0.3, 0.3],
        marker="s",
        c=rsites.color,
    )

    for i in range(len(x)):
        ax.annotate(f'{i} - {rsites.brain_region.iloc[i]}', (x[i]+.05, rsites.probe_coordinates.iloc[i]))

    ax.set(ylabel="Probe position (um)", xticks=[], xlim=[0.5, 1.5])
    plt.show()
