from pathlib import Path
import brainrender as br
from fcutils.path import files
from loguru import logger
import numpy as np

from myterial import green, grey_dark, grey, black, indigo

br.settings.SHOW_AXES = False
# br.set_logging('DEBUG')

probes_folder = Path(
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\reconstructed_probe_location"
)

colors = {
    '1100279_0': grey,
    '1100281_0': grey_dark,
    '1100516_0': indigo,
    '1110750_0': green,
    '1110751_0': black,
}

["k", "b", "r", "m", green]

# scene and brain regions
scene = br.Scene()

scene.add_brain_region("CUN", "GRN", alpha=.88)
scene.add_brain_region(
    "IC", "PRNr", "PRNc", "SCm", alpha=0.25, silhouette=False
)
scene.add_brain_region(
    "PPN",  alpha=0.8, silhouette=False
)
# add probes
for n, probe in enumerate(files(probes_folder, pattern="*_0.npy")):
    coords = np.load(probe)
    probe_actor = scene.add(br.actors.Points(coords, colors=colors[probe.stem], radius=40))
    logger.info(f"Adding probe from {probe.name} with color: {colors[probe.stem]}")


# slice
plane = scene.atlas.get_plane(
    norm=(0, 0, 1), pos=scene.root._mesh.centerOfMass()
)
scene.slice(plane)

# render
cam = {
    "pos": (7196, 247, -38602),
    "viewup": (0, -1, 0),
    "clippingRange": (29133, 44003),
    "focalPoint": (7718, 4290, -3507),
    "distance": 35331,
}
scene.render(camera=cam, zoom=2)
