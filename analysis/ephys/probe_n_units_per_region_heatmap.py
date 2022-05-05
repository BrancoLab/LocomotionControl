import sys
import pandas as pd
from rich import print
from rich.table import Table
from rich.box import MINIMAL

sys.path.append("./")
from data.dbase import db_tables


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
counts = (
    all_units.groupby("brain_region")
    .count()
    .sort_values("name", ascending=False)
)
counts = counts[["name"]]
counts.columns = ["# units"]
counts.append(counts.sum().rename("Total"))


tb = Table(box=MINIMAL, show_footer=True)
tb.add_column("Region", style="yellow", justify="right")
tb.add_column("Count", f"Tot: {counts['# units'].sum()}")

for r, n in counts.iterrows():
    tb.add_row(r, str(n.values[0]))
print(tb)


# ---------------------------------------------------------------------------- #
#                                    heatmap                                   #
# ---------------------------------------------------------------------------- #


import bgheatmaps as bgh

"""
    This example shows how to use visualize a heatmap in 2D
"""

values = dict(counts)


f = bgh.heatmap(
    values,
    position=0,  
    orientation="sagittal",  # 'frontal' or 'sagittal', or 'horizontal' or a tuple (x,y,z)
    title="cellcounts",
    vmin=-5,
    vmax=3,
    format="2D",
).show()