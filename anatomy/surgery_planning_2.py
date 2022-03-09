from brainrender.actors import Cylinder, Point
from brainrender import Scene, settings
import numpy as np
from vedo import shapes
from pathlib import Path

# To measure distances and angles
save_folder = Path(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
)

settings.SHOW_AXES = False

BREGMA = np.array([5400, 0, 0])  # AP  # DV  # ML

top = np.array([4.136, -2.106, 0]) * 1000 + BREGMA  # AP ML DV
tip = np.array([5.507, -0.584, 6.489]) * 1000 + BREGMA


scene = Scene(inset=False, screenshots_folder=save_folder,)
cun, grn, mos = scene.add_brain_region("CUN", "GRN", "MOs", alpha=0.1)
mos5 = scene.add_brain_region("MOs5", alpha=0.7)
orb, olf = scene.add_brain_region("ORB", "OLF")


# CUN/GRN probe
tip[1] = tip[1] + scene.root.centerOfMass()[2]
top[1] = top[1] + scene.root.centerOfMass()[2]

top = top[[0, 2, 1]]
tip = tip[[0, 2, 1]]

scene.add(shapes.Cylinder(pos=[top, tip], c="k", r=50, alpha=1))

BREGMA_centered = [
    5400,  # AP
    0,  # DV
    5700,  # ML
]

M2_probe_position = BREGMA_centered + np.array([-2500, 2500, -1000])
scene.add(Point(M2_probe_position, color="red", res=12, radius=150))

scene.add(Point(BREGMA_centered, color="blue"))

# MOs probe
mos_center = mos.centerOfMass() + np.array([1000, 0, -800])
for x in [0, 250, 500, 750]:
    scene.add(
        Cylinder(
            M2_probe_position + np.array([0, 0, -x]),
            scene.root,
            color="k",
            radius=75 / 2,
        )
    )

scene.slice(
    scene.atlas.get_plane(
        M2_probe_position + np.array([-100, 0, 0]), norm=(1, 0, 0)
    ),
    actors=[mos, scene.root, mos5, orb, olf],
)

scene.slice(
    scene.atlas.get_plane(
        M2_probe_position + np.array([1000, 0, 0]), norm=(-1, 0, 0)
    ),
)

camera = {
    "pos": (-6374, -5444, 26602),
    "viewup": (0, -1, 0),
    "clippingRange": (19433, 56931),
    "focalPoint": (7830, 4296, -5694),
    "distance": 36602,
}

scene.render(interactive=True, camera="frontal")
scene.screenshot(name="probes")
