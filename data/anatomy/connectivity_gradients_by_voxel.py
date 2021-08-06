import sys

sys.path.append('/Users/federicoclaudi/Documents/Github/MouseAnatomyViewer')
from aviewer.app import App

import brainrender as br

br.settings.vsettings.useDepthPeeling = False
br.settings.vsettings.alphaBitPlanes = 1
br.settings.vsettings.maxNumberOfPeels = 2
br.settings.vsettings.occlusionRatio = 0.1

br.settings.vsettings.screenshotTransparentBackground = False
br.settings.vsettings.useFXAA = True  # This needs to be false for transparent bg

cam =     {
    'pos': (-4767, -3533, -31261),
    'viewup': (0, -1, 0),
    'clippingRange': (22849, 40012),
    'focalPoint': (11035, 4324, -6224),
    'distance': 30632,
}
dest_fld = '/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/anatomy/connectivity_gradients'

source = 'SCm'
for voxel_idx in (980, -1000, 0, 500, 1200, 1500, 1800, 2000, 2400, 2800):
    app = App()
    
    app.add_projection(source, ['GRN',  'CUN', 'PPN'], th=.01, cmap='Reds', voxel_idx=voxel_idx, voxel_color='blue')


    # app.add_region_voxels("SCm")


    app.scene.add_brain_region('PAG', alpha=.1, silhouette=False)
    app.scene.render(interactive=False, zoom=2, camera=cam)
    app.scene.screenshot(dest_fld + f'/projections_from_{source}_voxel_{voxel_idx}')
    