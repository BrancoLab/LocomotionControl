import sys

sys.path.append("/Users/federicoclaudi/Documents/Github/MouseAnatomyViewer")
from aviewer.app import App

import brainrender as br

br.settings.vsettings.useDepthPeeling = False
br.settings.vsettings.alphaBitPlanes = 1
br.settings.vsettings.maxNumberOfPeels = 2
br.settings.vsettings.occlusionRatio = 0.1

br.settings.vsettings.screenshotTransparentBackground = False
br.settings.vsettings.useFXAA = (
    True  # This needs to be false for transparent bg
)


cam = {
    "pos": (-4767, -3533, -31261),
    "viewup": (0, -1, 0),
    "clippingRange": (22849, 40012),
    "focalPoint": (11035, 4324, -6224),
    "distance": 30632,
}


"""
    Plots gradients of connectivity from a bunch of brain regions to brain regions of interest
"""
dest_fld = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/anatomy/connectivity_gradients"
sources = [
    ("MOs", "Greens"),
    ("MOp", "Greens"),
    ("RSP", "Blues"),
    ("STR", "Blues"),
    ("PAG", "Purples"),
    ("ZI", "Reds"),
    ("SCm", "Reds"),
    ("CUN", "YlGnBu"),
    ("GRN", "YlOrRd"),
]

targets = [
    "SCm",
    "CUN",
    "PPN",
    "GRN",
]

for source, cmap in sources:
    app = App()

    app.add_projection(source, targets, th=0.1, cmap=cmap)

    app.scene.add_brain_region("PAG", alpha=0.1, silhouette=False)
    app.scene.render(interactive=False, zoom=2, camera=cam)
    app.scene.screenshot(
        dest_fld + f'/projections_from_{source}_to_{"_".join(targets)}'
    )

    # break
