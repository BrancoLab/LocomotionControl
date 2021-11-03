# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")


import matplotlib.pyplot as plt
import pandas as pd
import pathlib

from myterial.utils import make_palette
from tpd import recorder
from myterial import pink_dark, blue_dark

from control import trajectory_planning as tp
import draw
from data import paths
from data.data_structures import LocomotionBout
from data.data_utils import convolve_with_gaussian
from geometry import Path


folder = pathlib.Path(
    r"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
)
recorder.start(
    base_folder=folder.parent, folder_name=folder.name, timestamp=False
)


# %%
K = 11  # number of control points

(
    left_line,
    center_line,
    right_line,
    left_to_track,
    center_to_track,
    right_to_track,
    control_points,
) = tp.extract_track_from_image(
    points_spacing=2, k=K, apply_extra_spacing=True, restrict_extremities=True
)


# ? plot
f, ax = plt.subplots(figsize=(8, 12))


_ = draw.Hairpin()

P = [0, 0.125, 1]
colors = make_palette(pink_dark, blue_dark, len(P))
traces = []
for n, p in enumerate(P):
    trace, _ = tp.fit_best_trace(
        control_points,
        center_line,
        K,
        angle_cost=p,
        length_cost=(1 - p) * 5e-3,
    )

    # draw best trace
    X = convolve_with_gaussian(trace.x, kernel_width=11)
    Y = convolve_with_gaussian(trace.y, kernel_width=11)
    traces.append(Path(X, Y))
    draw.Tracking(X, Y, color=colors[n], lw=3)


# %%
# ------------------------------- load tracking ------------------------------ #
_bouts = pd.read_hdf(
    paths.analysis_folder / "behavior" / "saved_data" / f"complete_bouts.h5"
)
_bouts = _bouts.loc[_bouts.duration < 8]
print(f"Kept {len(_bouts)} bouts")

bouts = []
for i, bout in _bouts.iterrows():
    bouts.append(LocomotionBout(bout))

#  %%
# ----------------------------------- plot ----------------------------------- #
f, ax = plt.subplots(figsize=(8, 12))
_ = draw.Hairpin(set_ax=True)
f._save_name = "optimal_trajectory"

for bout in bouts:
    _ = draw.Tracking(bout.x + 0.5, bout.y - 0.5, alpha=0.2)

for trace, color in zip(traces, "brg"):
    draw.Tracking(trace.x, trace.y, lw=3, color=color)

recorder.add_figures()
# %%
