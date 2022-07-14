import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append("./")


from brainrender import Scene, settings
from brainrender.actors import Points
from data.dbase.db_tables import Probe
from myterial import blue_grey, grey_darker
from fcutils.path import files

settings.SHOW_AXES = False

TARGET = "CUN/GRN"

probes = (Probe).fetch(as_dict=True)


save_fld = Path("D:\Dropbox (UCL)\Rotation_vte\Writings\THESIS\Chpt4\Plots")
scene = Scene(screenshots_folder=save_fld)

regions = ("MOs", "CUN", "PPN")
regions_meshes = scene.add_brain_region(*regions, alpha=0.3, silhouette=False)
# scene.slice(plane="frontal", actors=[scene.root])

for probe in probes:
    # get and visualize the probe from the reconstruction file.
    mouse = probe["mouse_id"][-3:]
    if mouse == "750":
        continue

    if probe["target"] != TARGET:
        continue

    if probe["target"] == "MOs":
        mouse_files = files(
            Path(
                r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\reconstructed_probe_location"
            )
            / mouse
        )

        probe_points = np.vstack([np.load(f) for f in mouse_files])
        CONFIGURATION = "r96"
    else:
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
        CONFIGURATION = "b0"

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

    if TARGET == "CUN/GRN":
        rsites = rsites.loc[rsites.brain_region.isin(regions)]

    track = np.vstack(rsites.registered_brain_coordinates.values)
    colors = [
        color
        if region in regions  # or "MOs" in region
        else (blue_grey if region not in ("unknown", "OUT") else "k")
        for color, region in zip(
            rsites.color.values, rsites.brain_region.values
        )
    ]
    pts = scene.add(Points(track, colors=colors, radius=30))
    scene.add_silhouette(pts, lw=2)
    # scene.render(interactive=True)

scene.render(camera="frontal", interactive=False, zoom=1.25)
tgt = TARGET.replace("/", "_")
scene.screenshot(name=f"probes_renderings_{tgt}_1")

scene.render(camera="sagittal", interactive=False, zoom=1.5)
scene.screenshot(name=f"probes_renderings_{tgt}_2")
scene.render(interactive=False, zoom=0.8)
scene.screenshot(name=f"probes_renderings_{tgt}_3")

scene.close()
