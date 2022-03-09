# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from tpd import recorder

from data import paths
from data.data_structures import LocomotionBout
from kinematics import track
from kinematics import track_cordinates_system as TCS

import draw

"""
    Plots complete bouts through the arena, looking at linearized track
    quantities (e.g. speed vs track position)
"""

folder = Path(
    r"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
)
recorder.start(
    base_folder=folder.parent, folder_name=folder.name, timestamp=False
)


# %%
# load and clean complete bouts
_bouts = pd.read_hdf(
    paths.analysis_folder / "behavior" / "saved_data" / f"complete_bouts.h5"
)
_bouts = _bouts.loc[_bouts.start_roi == 0]
_bouts = _bouts.sort_values("duration").iloc[:10]
print(f"Kept {len(_bouts)} bouts")

bouts = []
for i, bout in _bouts.iterrows():
    bouts.append(LocomotionBout(bout))

# %%
(
    left_line,
    center_line,
    right_line,
    left_to_track,
    center_to_track,
    right_to_track,
    control_points,
) = track.extract_track_from_image(
    points_spacing=1, restrict_extremities=False, apply_extra_spacing=False,
)


# %%
f = plt.figure(figsize=(12, 12))
axes = f.subplot_mosaic(
    """
        AAADDD
        AAAEEE
        AAAFFF
        BBBGGG
        CCCHHH
    """
)
f.tight_layout()

_ = draw.Hairpin(ax=axes["A"])
for bout in bouts:
    # draw 2d and linearized tracking
    draw.Tracking(bout.x, bout.y, ax=axes["A"])

    linearized = TCS.path_to_track_coordinates_system(center_line, bout.path)
    draw.Tracking(linearized.x, linearized.y, ax=axes["B"])

    # draw speed accel and ang vel along track
    axes["D"].plot(linearized.x, bout.speed, color="k", alpha=0.5)
    axes["E"].plot(linearized.x, bout.acceleration, color="k", alpha=0.5)

    axes["F"].plot(linearized.x, bout.thetadot, color="k", alpha=0.5)
    axes["G"].plot(linearized.x, bout.thetadotdot, color="k", alpha=0.5)

    # draw distance along the track for each path
    axes["H"].plot(linearized.x, color="k", alpha=0.5)


draw.draw_track(center_line, left_line, right_line, ax=axes["A"])

draw.draw_track(center_to_track, left_to_track, right_to_track, ax=axes["B"])

axes["C"].plot(center_line.comulative_distance, center_line.curvature)

for ax in "EFG":
    axes[ax].axhline(0, lw=2, color="k", zorder=-1)


axes["B"].set(tutle="linearized track")
axes["C"].set(tutle="linearized track curvature")
axes["D"].set(tutle="speed")
axes["E"].set(tutle="acceleration")
axes["F"].set(tutle="angular velocity")
axes["G"].set(tutle="angular acceleration")
axes["H"].set(tutle="track position")

# %%
