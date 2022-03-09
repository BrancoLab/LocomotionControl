from pathlib import Path

import brainrender as br
from brainrender.atlas_specific import get_streamlines_for_region
from brainrender.actors.streamlines import make_streamlines

SELECTED = "locomotion"


save_folder = Path(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
)


br.settings.SHOW_AXES = False
br.settings.vsettings.screenshotTransparentBackground - True


scene = br.Scene(screenshots_folder=save_folder, inset=False)

scene.add_brain_region("MOs", alpha=1)
scene.add_brain_region("CUN", "PPN", "GRN", alpha=0.3)

streams = get_streamlines_for_region("MOs")[2:5]
scene.add(
    *make_streamlines(
        *streams,
        color=scene.atlas._get_from_structure("MOs", "rgb_triplet"),
        alpha=0.5,
    )
)

camera = {
    "pos": (-6374, -5444, 26602),
    "viewup": (0, -1, 0),
    "clippingRange": (19433, 56931),
    "focalPoint": (7830, 4296, -5694),
    "distance": 36602,
}

scene.render(interactive=False, camera=camera)
scene.screenshot(name="mos_to_loc")
