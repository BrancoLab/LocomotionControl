import sys
import pandas as pd
import numpy as np

from pathlib import Path
from brainrender import Scene, settings
from brainrender.actors import Points

sys.path.append("./")
from data.dbase import db_tables


settings.BACKGROUND_COLOR = "white"
settings.SHOW_AXES = False

save_fld = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys")


"""
    Gets the total number of units for each brain region across all recordings.
    It prints the count in a nicely formatted table + creates a heatmap to display the numbers.
"""

# ---------------------------------------------------------------------------- #
#                                get/count units                               #
# ---------------------------------------------------------------------------- #
recordings = (db_tables.Recording).fetch(as_dict=True)
all_units = []
for recording in recordings:
    cf = recording["recording_probe_configuration"]
    units = db_tables.Unit.get_session_units(
        recording["name"],
        cf,
        spikes=True,
        firing_rate=False,
        frate_window=100,
    )
    units["probe_configuration"] = [cf] * len(units)
    units["recording"] = [recording["mouse_id"]] * len(units)

    units_regions = []
    for i, unit in units.iterrows():
        if "RSP" in unit.brain_region:
            units_regions.append("RSP")
        elif "VISp" in unit.brain_region:
            units_regions.append("VISp")
        else:
            units_regions.append(unit.brain_region)

    units["brain_region"] = units_regions

    rsites = pd.DataFrame(
        (
            db_tables.Probe.RecordingSite
            & recording
            & f'probe_configuration="{cf}"'
        ).fetch()
    )

    if not len(units):
        continue

    all_units.append(units)

all_units = pd.concat(all_units)
counts_by_region = (
    all_units.groupby("brain_region")
    .count()
    .sort_values("name", ascending=False)
)


REGION = "CUN"
units = all_units.loc[all_units.brain_region == REGION]

scene = Scene()
scene.add_brain_region(REGION, alpha=0.7)

coords = np.vstack(units.registered_brain_coordinates.values)
cells = scene.add(Points(coords, colors="black", radius=15))
scene.add_silhouette(cells)

scene.render()
a = 1
