import sys
import pandas as pd
from rich import print
from rich.table import Table
from rich.box import MINIMAL

import bgheatmaps as bgh
from pathlib import Path
import brainrender
from myterial import blue_light, blue_lighter

sys.path.append("./")
from data.dbase import db_tables


brainrender.settings.BACKGROUND_COLOR = "white"

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

    # rsites = pd.DataFrame(
    #     (
    #         db_tables.Probe.RecordingSite
    #         & recording
    #         & f'probe_configuration="{cf}"'
    #     ).fetch()
    # )

    if not len(units):
        continue

    all_units.append(units)

all_units = pd.concat(all_units)
counts = all_units.groupby(["mouse_id", "brain_region"]).size()

mice = sorted(all_units.mouse_id.unique())
regions = sorted(all_units.brain_region.unique())

counts_by_region = (
    all_units.groupby("brain_region")
    .count()
    .sort_values("name", ascending=False)
)
counts_by_region = counts_by_region[["name"]]
counts_by_region.columns = ["# units"]
counts_by_region.append(counts_by_region.sum().rename("Total"))

tb = Table(box=MINIMAL, show_footer=True)
tb.add_column("MOUSE", style="yellow", justify="right")
for region in regions:
    tot_units = counts_by_region.loc[counts_by_region.index == region][
        "# units"
    ][0]
    tb.add_column(
        region,
        f"[dim]Tot:[/dim][bold] {tot_units}[/]",
        header_style="bold green",
        justify="center",
    )

for mouse in mice:
    row = [mouse]
    mouse_units = counts[mouse]

    for region in regions:
        if region in mouse_units.index:
            row.append(str(mouse_units[region]))
        else:
            row.append("")

    tb.add_row(*row)
print(tb)


# tb = Table(box=MINIMAL, show_footer=True)
# tb.add_column("Region", style="yellow", justify="right")
# tb.add_column("Count", f"Tot: {counts['# units'].sum()}")

# for r, n in counts.iterrows():
#     tb.add_row(r, str(n.values[0]))
# print(tb)


# ---------------------------------------------------------------------------- #
#                                    heatmap                                   #
# ---------------------------------------------------------------------------- #


values = {r: int(c.values[0]) for r, c in counts_by_region.iterrows()}

for key in ("OUT", "unknown", "bic", "tb", "scwm"):
    if key in values.keys():
        del values[key]


positions = (4250, 5250)


scene = brainrender.Scene(screenshots_folder=save_fld)
com = scene.root._mesh.centerOfMass()
scene.root.alpha(1)
colors = (blue_light, blue_lighter)
for pos, color, alpha in zip(positions, colors, (0.75, 9)):
    p = com
    p[2] = -p[2]
    p[2] = pos
    plane = scene.atlas.get_plane(
        pos=p, norm=(0, 0, 1), sx=18000, sy=10000, alpha=alpha, color=color
    )
    plane = scene.add(plane)
    scene.add_silhouette(plane)

    intersection = scene.root.mesh.intersectWith(plane).lineWidth(4).c("black")
    scene.add(intersection)


scene.render(
    camera={
        "pos": (-40685, -11861, -22445),
        "viewup": (0, -1, 0),
        "clippingRange": (30733, 82928),
        "focalPoint": (7553, 4657, -5791),
        "distance": 53213,
    },
    #    camera="frontal",
    interactive=False,
)
scene.screenshot(name="n_units_slicing_planes")
scene.close()
del scene

for pos in positions:
    f = bgh.heatmap(
        values,
        position=pos,
        orientation="sagittal",  # 'frontal' or 'sagittal', or 'horizontal' or a tuple (x,y,z)
        title="cellcounts",
        # vmin=0,
        # vmax=354,
        thickness=300,
        format="2D",
        label_regions=False,
    ).plot(cbar_label="# cells", save_fld=True)

    f.savefig(save_fld / f"n_units_heatmap_{pos}.png", dpi=600)
