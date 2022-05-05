import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append("./")


from brainrender import Scene
from brainrender.actors import Points
from data.dbase.db_tables import Probe, Session, ValidatedSession

CONFIGURATION = "longcol"

probes = (Probe * Session * ValidatedSession).fetch()
scene = Scene()

for probe in probes:
    # get and visualize the probe from the reconstruction file.
    mouse = probe["mouse_id"][-3:]
    rec_file = Path(
        r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\reconstructed_probe_location"
    ).glob(mouse + "_atlas_space_0.npy")[0]

    probe_points = np.load(rec_file)[
                ::-1
            ]  # flipped so that the first point is at the bottom of the probe like in brain
    scene.add(Points(probe_points, colors="black", radius=15))

    # get and visualize the probe's recording sites
    mouse = probe["mouse_id"]
    rsites = pd.DataFrame(
        (
            Probe.RecordingSite
            & f'mouse_id="{mouse}"'
            & f'probe_configuration="{CONFIGURATION}"'
        ).fetch()
    )

    track = np.vstack(rsites.registered_brain_coordinates.values)
    colors = [
        color
        # if region in targets
        # else (blue_grey if region not in ("unknown", "OUT") else "k")
        for color, region in zip(
            rsites.color.values, rsites.brain_region.values
        )
    ]
    pts = scene.add(Points(track, colors=colors, radius=15))
    scene.add_silhouette(pts, lw=2)


scene.render()
