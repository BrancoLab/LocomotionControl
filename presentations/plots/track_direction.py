# %%
import sys
import pathlib

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import matplotlib.pyplot as plt
from tpd import recorder
from control import trajectory_planning as tp
import draw

folder = pathlib.Path(
    r"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
)
recorder.start(
    base_folder=folder.parent, folder_name=folder.name, timestamp=False
)

#  %%
(
    left_line,
    center_line,
    right_line,
    left_to_track,
    center_to_track,
    right_to_track,
    control_points,
) = tp.extract_track_from_image(points_spacing=1)

center_line = center_line.downsample(4)
# %%
f, ax = plt.subplots(figsize=(8, 12))
f._save_name = "track_direction"

# draw.Tracking(center_line.x, center_line.y, lw=3)
from myterial.utils import make_palette
from myterial import pink_dark, blue

colors = make_palette(pink_dark, blue, len(center_line))

draw.Arrows(
    center_line.x,
    center_line.y,
    center_line.tangent.angle,
    step=1,
    L=3,
    width=4,
    outline=True,
    color=colors,
)

_ = draw.Hairpin()


recorder.add_figures()
