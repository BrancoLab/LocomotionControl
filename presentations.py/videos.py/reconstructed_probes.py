from pathlib import Path
from typing_extensions import TypeVarTuple

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
# scene and brain regions
scene = br.Scene(inset=False, screenshots_folder=save_folder)

scene.add_brain_region("CUN", "GRN", "PPN", alpha=.9)


# add probes
for n, probe in enumerate(probes_files):
    coords = np.load(probe)
    probe_actor = scene.add(br.actors.Points(coords[::4, :], colors=colors[n], radius=60))
    # scene.add_silhouette(probe_actor, lw=.5)


# render
scene.render(camera=None, zoom=1.5, interactive=TypeVarTuple)
# scene.screenshot(name=f'implanted_probes_{name}')