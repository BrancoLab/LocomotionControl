import brainrender as br
import numpy as np
import pandas as pd
from pathlib import Path

highlighterd = [
    'CUN', 'GRN', 'PRNc', 'PRNr', 'MLR', 'PPN', 'SCm'
]

br.settings.SHOW_AXES = False

def render_probe_3d(rsites:pd.DataFrame, save_path:Path=None):

    # create brainrender scene
    scene = br.Scene(screenshots_folder=save_path)

    # add probe track
    track = np.vstack(rsites.registered_brain_coordinates.values)
    scene.add(br.actors.Points(track, colors='k', radius=80))

    # add brain regions
    for region in rsites.brain_region.unique():
        alpha = .7 if region in highlighterd else .05
        
        actor = scene.add_brain_region(region, alpha=alpha)

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

    if save_path is None:
        scene.render(camera=cam)
    else:
        scene.render(camera=cam, interactive=False)
        scene.screenshot('probe_rendering')






if __name__ == '__main__':
    import sys
    sys.path.append('./')

    from data.dbase.db_tables import Probe

    rsites = pd.DataFrame((Probe.RecordingSite & 'mouse_id="AAA1110750"').fetch())

    render_probe_3d(rsites)