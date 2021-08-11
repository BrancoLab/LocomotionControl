import sys

sys.path.append("/Users/federicoclaudi/Documents/Github/BrainRender")

from brainrender import Scene, settings
from brainrender.atlas_specific import get_streamlines_for_region
from brainrender.actors.streamlines import make_streamlines
import brainrender as br

br.set_logging("DEBUG")

dest_fld = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/anatomy/streamlines"

settings.SHOW_AXES = False


region = "SCm"

regions = ("SCm", "CUN", "PPN", "GRN")


cam = {
    "pos": (-4767, -3533, -31261),
    "viewup": (0, -1, 0),
    "clippingRange": (22849, 40012),
    "focalPoint": (11035, 4324, -6224),
    "distance": 30632,
}


# Create a brainrender scene
scene = Scene(title="projections from " + region, root=False)

# Add brain regions
scene.add_brain_region(*regions, alpha=0.6, silhouette=True)
scene.add_brain_region("PAG", alpha=0.2, silhouette=False)

# Get stramlines data and add
streams = get_streamlines_for_region(region)

zoom = 2
for n, stream in enumerate(streams):
    act = scene.add(*make_streamlines(stream, color="salmon", alpha=0.6))

    # Render!
    scene.render(interactive=False, zoom=zoom, camera=cam)
    scene.screenshot(dest_fld + f"/{region}_streams_{n}")

    scene.remove(act)
    zoom = 1

    # break
