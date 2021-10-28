from pathlib import Path

from loguru import logger
import numpy as np

from myterial.utils import make_palette
import brainrender as br
from myterial import blue_grey_dark, blue_grey_light
from fcutils.path import files


# ----------------------------------- prep ----------------------------------- #
br.settings.SHOW_AXES = False
br.settings.ROOT_ALPHA = .1

probes_folder = Path(
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\reconstructed_probe_location"
)
probes_files = files(probes_folder, pattern="*_0.npy")

save_folder = Path(
    r"D:\Dropbox (UCL)\Rotation_vte\Presentations\Presentations\Fiete lab"
)


colors = make_palette(blue_grey_dark, blue_grey_light, len(probes_files))



# ----------------------------- create renderings ---------------------------- #
for repeat in range(3):
    # scene and brain regions
    scene = br.Scene(inset=False, screenshots_folder=save_folder)

    scene.add_brain_region("CUN", "GRN", "PPN", alpha=.9)


    # add probes
    for n, probe in enumerate(probes_files):
        coords = np.load(probe)
        probe_actor = scene.add(br.actors.Points(coords[::4, :], colors=colors[n], radius=60))
        # scene.add_silhouette(probe_actor, lw=.5)

    # slice
    if repeat == 2:
        plane = scene.atlas.get_plane(
            norm=(0, 0, -1), pos=scene.root._mesh.centerOfMass()
        )
        camera = 'sagittal'
        name = 'side'
    elif repeat == 1:
        plane = scene.atlas.get_plane(
            norm=(0, 1, 0), pos=scene.root._mesh.centerOfMass()
        )
        camera = {
        'pos': (7742, -39542, -5586),
        'viewup': (0, 0, -1),
        'clippingRange': (35999, 53205),
        'focalPoint': (7654, 5943, -5267),
        'distance': 45486,
        }
        name = 'top'
    else:
        name = 'angled'
        plane = None
        camera = None

    if plane is not None:
        scene.slice(plane, actors=[scene.root])

    # render
    scene.render(camera=camera, zoom=1.5, interactive=False)
    scene.screenshot(name=f'implanted_probes_{name}')