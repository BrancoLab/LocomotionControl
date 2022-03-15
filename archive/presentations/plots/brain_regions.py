from pathlib import Path

import brainrender as br


SELECTED = "space"


save_folder = Path(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
)


br.settings.SHOW_AXES = False
br.settings.vsettings.screenshotTransparentBackground - True
camera = {
    "pos": (-6374, -5444, 26602),
    "viewup": (0, -1, 0),
    "clippingRange": (19433, 56931),
    "focalPoint": (7830, 4296, -5694),
    "distance": 36602,
}


regions = dict(
    space=("CA", "ENT", "ORB"),
    action=("RSP",),
    motor=("MOs",),
    locomotion=("CUN", "GRN", "PPN"),
)

for region in regions[SELECTED]:
    scene = br.Scene(screenshots_folder=save_folder, inset=False)

    scene.add_brain_region(region)

    scene.render(interactive=False, camera=camera)
    scene.screenshot(name=region)