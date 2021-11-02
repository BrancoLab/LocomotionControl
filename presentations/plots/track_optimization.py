# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from myterial.utils import make_palette
from myterial import pink_dark, blue_dark

from control import trajectory_planning as tp
import draw
from data import paths
from data.data_structures import LocomotionBout


# %%
N = 10  # number of traces
K = 9  # number of control points

(
    left_line,
    center_line,
    right_line,
    left_to_track,
    center_to_track,
    right_to_track,
    control_points,
) = tp.extract_track_from_image(points_spacing=3, k=K)

min_len_trace_path, min_len_trace_to_track = tp.fit_best_trace(
    control_points, center_line, K, angle_cost=0, length_cost=1
)
min_ang_trace_path, min_ang_trace_to_track = tp.fit_best_trace(
    control_points, center_line, K, angle_cost=1, length_cost=0
)

# ? plot
f, ax = plt.subplots(figsize=(8, 12))


_ = draw.Hairpin()


colors = make_palette(pink_dark, blue_dark, N)
for n, p in enumerate(np.linspace(0, 1, N)):
    trace, _ = tp.fit_best_trace(
        control_points,
        center_line,
        K,
        angle_cost=p,
        length_cost=(1 - p) * 2.5e-2,
    )

    # draw best trace
    draw.Tracking(trace.x, trace.y, color=colors[n], lw=3)


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
_ = draw.Hairpin()

for bout in bouts:
    _ = draw.Tracking(bout.x + 1, bout.y, alpha=0.2)
