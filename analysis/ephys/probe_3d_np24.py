import numpy as np
from pathlib import Path


from myterial import blue_dark, indigo_dark
from brainrender import Scene, settings
from brainrender.actors import Points

settings.SHOW_AXES = False

folders = [
    Path(
        r"W:\swc\branco\BrainSaw\YI_0000012b\brainreg\manual_segmentation\sample_space\tracks"
    ),
    Path(
        r"W:\swc\branco\BrainSaw\YI_1101192\brainreg\manual_segmentation\sample_space\tracks"
    ),
]
colors = (blue_dark, indigo_dark)

scene = Scene(
    screenshots_folder=r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\ephys"
)
mos = scene.add_brain_region("MOs", alpha=0.25, silhouette=False)
mos5 = scene.add_brain_region("MOs5", alpha=0.8)

for folder, color in zip(folders, colors):
    for fpath in folder.glob("*.npy"):
        points = np.load(fpath)
        pts = scene.add(Points(points, colors=color))
        scene.add_silhouette(pts)

# side view
# plane = scene.atlas.get_plane(norm=(0, 0, 1))
# scene.slice(plane, actors=[scene.root, mos, mos5])
# scene.render(camera="sagittal2", interactive=False)
# scene.screenshot("probes_np24_sideview")

# front view
# plane = scene.atlas.get_plane((2800, 0, 0), norm=(1, 0, 0))
# scene.slice(plane, actors=[scene.root])
# plane = scene.atlas.get_plane((2100, 0, 0), norm=(1, 0, 0))
# scene.slice(plane, actors=[mos, mos5])
# scene.render(camera="frontal", interactive=True)
# scene.screenshot("probes_np24_frontview")

scene.screenshot("probes_np24_defaultview")
