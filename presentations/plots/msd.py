# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tpd import recorder
from fcutils.plot.elements import plot_mean_and_error

from data import paths
from data.data_structures import LocomotionBout
import draw
from data.dbase._tracking import calc_angular_velocity
from kinematics import MSD

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=False)

folder = Path(
    r"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
)
recorder.start(
    base_folder=folder.parent, folder_name=folder.name, timestamp=False
)

"""
    Fit MSD model to turn trajectories in mice
"""

# %%
# load and clean roi crossings
ROI = "T2"
MIN_DUR = 1.5

_bouts = pd.read_hdf(
    paths.analysis_folder / "behavior" / "saved_data" / f"{ROI}_crossings.h5"
).sort_values("duration")

_bouts = _bouts.loc[_bouts.duration < MIN_DUR]


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
N = 10
S, T = np.full((80, N), np.nan), np.full((80, N), np.nan)
_S, _T = np.full((80, N), np.nan), np.full((80, N), np.nan)

for i in range(N):
    bout = LocomotionBout(_bouts.iloc[i])
    trajectory, time = MSD(
        bout, skip=SKIP, start_frame=3, end_frame=23
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

    # store tracking data
    S[: len(bout), i] = bout.speed
    T[: len(bout), i] = bout.thetadot

    # store simulation results
    _S[time[0] : time[-1], i] = trajectory.speed
    _T[time[0] : time[-1], i] = calc_angular_velocity(trajectory.theta) * 60

_T[:7, :] = np.nan

plot_mean_and_error(np.nanmedian(S, 1), np.nanstd(S, 1), axes["B"])

plot_mean_and_error(
    np.nanmedian(_S, 1), np.nanstd(_S, 1), axes["B"], color="red"
)

plot_mean_and_error(np.nanmedian(T, 1), np.nanstd(T, 1), axes["C"])
plot_mean_and_error(
    np.nanmedian(_T, 1), np.nanstd(_T, 1), axes["C"], color="red"
)


# %%

# %%

m = MSD(bout, skip=SKIP, start_frame=3, end_frame=23)
trajectory, _ = m.simulate()

# plt.plot(trajectory.x, trajectory.y)
# plt.scatter(m.fits['x'].x_0, m.fits['y'].x_0)
# plt.scatter(m.fits['x'].x_1, m.fits['y'].x_1)

# plt.plot(trajectory.velocity.x)
# plt.scatter(0, m.fits['x'].v_0)
# plt.scatter(len(trajectory), m.fits['x'].v_1)

plt.plot(trajectory.velocity.y)
plt.scatter(0, m.fits["y"].v_0)
plt.scatter(len(trajectory), m.fits["y"].v_1)
# %%
