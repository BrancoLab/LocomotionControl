# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from tpd import recorder
from fcutils.plot.elements import plot_mean_and_error
from fcutils.progress import track

from data import paths
from data.data_structures import LocomotionBout
import draw
from data.dbase._tracking import calc_angular_velocity
from kinematics.msd import MSD
from kinematics import time
from geometry import Path

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=False)

folder = Path(
    # r"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
    r"D:\Dropbox (UCL)\Rotation_vte\Presentations\Presentations\Fiete lab"
)
recorder.start(
    base_folder=folder.parent, folder_name=folder.name, timestamp=False
)

"""
    Fit MSD model to turn trajectories in mice
"""

# %%
# load and clean roi crossings
ROI = "T3"
MIN_DUR = 1.3

_bouts = pd.read_hdf(
    paths.analysis_folder / "behavior" / "saved_data" / f"{ROI}_crossings.h5"
).sort_values("duration")

_bouts = _bouts.loc[_bouts.duration < MIN_DUR]
print(f"Kept {len(_bouts)} bouts")

# create bouts after resampling in time
paths = time.time_rescale(
    [Path(bout.x, bout.y).smooth() for i, bout in _bouts.iterrows()]
)


# %%
f = plt.figure(figsize=(16, 8))
axes = f.subplot_mosaic(
    """
        AABBB
        AACCC
    """
)

draw.ROI(ROI, ax=axes["A"])

SKIP = 0
N = 10  # len(_bouts)
S, T = np.full((80, N), np.nan), np.full((80, N), np.nan)
_S, _T = np.full((80, N), np.nan), np.full((80, N), np.nan)

simulated = []
for i in track(range(N)):
    bout = paths[i]
    trajectory, _time = MSD(
        bout, skip=SKIP, start_frame=2, end_frame=-2
    ).simulate()

    # plot
    draw.Tracking(bout.x, bout.y, zorder=-1, ax=axes["A"], alpha=0.8)
    draw.Tracking(
        trajectory.x,
        trajectory.y,
        zorder=10,
        ax=axes["A"],
        color="red",
        alpha=0.8,
    )
    simulated.append(trajectory)

    # store tracking data
    S[: len(bout), i] = bout.speed

    # store simulation results
    _S[_time[0] : _time[-1], i] = trajectory.speed

plot_mean_and_error(np.nanmedian(S, 1), np.nanstd(S, 1), axes["B"])

plot_mean_and_error(
    np.nanmedian(_S, 1), np.nanstd(_S, 1), axes["B"], color="red"
)


# %%
"""
    Plot average traces
"""
f = plt.figure(figsize=(16, 8))
axes = f.subplot_mosaic(
    """
        AABBB
        AACCC
    """
)

draw.ROI(ROI, ax=axes["A"])

for i in track(range(N)):
    bout = paths[i]

    draw.Tracking(bout.x, bout.y, zorder=-1, ax=axes["A"], alpha=0.8)

avg = time.average_xy_trajectory(paths)
draw.Tracking(avg.x, avg.y, lw=4, color="k", ax=axes["A"])


avg_sim = time.average_xy_trajectory(simulated)
draw.Tracking(avg_sim.x, avg_sim.y, lw=4, color="r", ax=axes["A"])
# %%
