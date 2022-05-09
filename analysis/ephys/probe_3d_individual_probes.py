import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append("./")


from brainrender import Scene, settings
from brainrender.actors import Points
from data.dbase.db_tables import Probe
from myterial import blue_grey, grey_darker

settings.SHOW_AXES = False

CONFIGURATION = "longcolumn"

probes = (Probe).fetch()


save_fld = Path("D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\analysis\\ephys")
regions = ["CUN", "PPN", "MRN", "PRNr", "PB"]

for probe in probes[1:]:
    mouse = probe["mouse_id"][-3:]

    scene = Scene(screenshots_folder=save_fld)

    regions_meshes = scene.add_brain_region(
        *regions, alpha=0.3, silhouette=False
    )
    # scene.slice(plane="frontal", actors=[scene.root])

    # get and visualize the probe from the reconstruction file.
    rec_file = list(
        Path(
            r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\reconstructed_probe_location"
        ).glob(mouse + "_atlas_space_0.npy")
    )

    if len(rec_file) == 0:
        continue

    probe_points = np.load(rec_file[0])[
        ::-1
    ]  # flipped so that the first point is at the bottom of the probe like in brain
    scene.add(Points(probe_points[::5], colors=grey_darker, radius=15))

    # get and visualize the probe's recording sites
    mouse = probe["mouse_id"]
    rsites = pd.DataFrame(
        (
            Probe.RecordingSite
            & f'mouse_id="{mouse}"'
            & f'probe_configuration="{CONFIGURATION}"'
        ).fetch()
    )
    # rsites = rsites.loc[rsites.brain_region.isin(regions)]

    track = np.vstack(rsites.registered_brain_coordinates.values)
    colors = [
        color
        if region in regions
        else (blue_grey if region not in ("unknown", "OUT") else "k")
        for color, region in zip(
            rsites.color.values, rsites.brain_region.values
        )
    ]
    pts = scene.add(Points(track, colors=colors, radius=30))
    scene.add_silhouette(pts, lw=2)

    scene.render(camera="frontal", interactive=False, zoom=1.5)
    scene.screenshot(name=f"probe_{mouse}_3d_1")

    scene.render(camera="sagittal", interactive=False, zoom=1.5)
    scene.screenshot(name=f"probe_{mouse}_3d_2")
    scene.render(zoom=1.0, interactive=False)
    scene.screenshot(name=f"probe_{mouse}_3d_3",)

    scene.close()
    del scene
