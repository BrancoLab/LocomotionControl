# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from scipy.stats import ttest_ind

from tpd import recorder
from myterial import pink, blue_dark, blue_grey_dark

from data import paths
from data.data_structures import LocomotionBout, merge_locomotion_bouts
import draw
from geometry import Vector

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
    Gets the tracking data of a ROI crossing and checks when the accelration
    gets below a given threhsold -> slowing down and if that's before when the
    angular acceleration > threshold (turning) to see if mice slow down in anticipation
"""

# %%
# ---------------------------------------------------------------------------- #
#                                   data prep                                  #
# ---------------------------------------------------------------------------- #
# load and clean roi crossings
ROI = "T3"
MIN_DUR = 1.5
USE = -1

_bouts = pd.read_hdf(
    paths.analysis_folder / "behavior" / "saved_data" / f"{ROI}_crossings.h5"
).sort_values("duration")

_l = len(_bouts)
_bouts = _bouts.loc[_bouts.duration <= MIN_DUR]
_bouts = _bouts.iloc[:USE]
print(f"Kept {len(_bouts)}/{_l} bouts")

crosses = []
for i, bout in _bouts.iterrows():
    crosses.append(LocomotionBout(bout))

# merge bouts for heatmaps
X, Y, S, A, T, AV, AA = merge_locomotion_bouts(crosses)


# ? Get when acceleration and angular accel. surpass thresholds for change
shift = 0
if ROI == "T2":
    ang_th = 6
elif ROI == "T1":
    ang_th = 30
    shift = 10
elif ROI == "T3":
    ang_th = -30
elif ROI == "T4":
    ang_th = -10
else:
    raise ValueError

th_crossess = dict(n=[], variable=[], frame=[], time=[], x=[], y=[])
for n, cross in enumerate(crosses):
    acc = np.where(cross.acceleration < -0.5)[0][0]
    try:
        if ROI in ("T1", "T2"):
            aac = np.where(cross.thetadotdot[shift:] > ang_th)[0][0] + shift
        elif ROI in ("T3", "T4"):
            aac = np.where(cross.thetadotdot[shift:] < ang_th)[0][0] + shift
        else:
            raise ValueError
    except IndexError:
        print("skipping one")
        continue

    for var, val in zip(("acceleration", "angular_acceleration"), (acc, aac)):
        th_crossess["variable"].append(var)
        th_crossess["time"].append(val / 60)
        th_crossess["frame"].append(val)
        th_crossess["x"].append(cross.x[val])
        th_crossess["y"].append(cross.y[val])
        th_crossess["n"].append(n)
th_crossess = pd.DataFrame(th_crossess)

# %%
# ---------------------------------------------------------------------------- #
#                          first plot - sanity checks                          #
# ---------------------------------------------------------------------------- #
"""
    Plot tracking over threshold crossings for sanity checks
"""
f = plt.figure(figsize=(24, 10))
axes = f.subplot_mosaic(
    """
        AAABBBBFFFG
        AAACCCCFFFH
    """
)

draw.ROI(
    ROI, ax=axes["A"], set_ax=True,
)
draw.ROI(
    ROI, ax=axes["F"], set_ax=True,
)

for n, cross in enumerate(crosses):
    time = np.linspace(0, cross.duration, len(cross.x))
    # mark when the thresholds were exceeded
    try:
        acc_cross = th_crossess.loc[
            (th_crossess.variable == "acceleration") & (th_crossess.n == n)
        ].iloc[0]
        aac_crooss = th_crossess.loc[
            (th_crossess.variable == "angular_acceleration")
            & (th_crossess.n == n)
        ].iloc[0]
    except IndexError:
        continue

    axes["B"].plot(time, cross.acceleration, color=blue_dark, alpha=0.2)
    axes["C"].plot(time, cross.thetadotdot, color=pink, alpha=0.2)

    axes["B"].scatter(
        time[acc_cross.frame],
        cross.acceleration[acc_cross.frame],
        color="k",
        alpha=1,
        zorder=100,
    )
    axes["C"].scatter(
        time[aac_crooss.frame],
        cross.thetadotdot[aac_crooss.frame],
        color="k",
        alpha=1,
        zorder=100,
    )

    axes["G"].plot(
        [0, 1], [acc_cross.time, aac_crooss.time], "-o", color="k", alpha=0.5
    )

    axes["A"].scatter(acc_cross.x, acc_cross.y, color="k", zorder=100)
    axes["F"].scatter(aac_crooss.x, aac_crooss.y, color="k", zorder=100)


# draw acceleration heatmap
draw.Tracking.heatmap(
    X, Y, c=A, gridsize=(40, 60), ax=axes["A"], vmin=-3, vmax=3, cmap="bwr"
)

# draw angular acceleration heatmap
draw.Tracking.heatmap(
    X + 0.5,
    Y + 0.5,
    c=AA,
    gridsize=(40, 60),
    ax=axes["F"],
    vmin=-30,
    vmax=30,
    cmap="PRGn",
)


# draw.Tracking.scatter(
#     path.x[0], path.y[0], color="k", zorder=200, ax=axes["F"]
# )
_ = axes["B"].axhline(0)

_ = axes["C"].axhline(0)

# %%


"""
    Plot the time between the threshold changes of the two variables and the distance between them
"""
# ---------------------------------------------------------------------------- #
#                            figure for presentation                           #
# ---------------------------------------------------------------------------- #

f = plt.figure(figsize=(20, 10))
axes = f.subplot_mosaic(
    """
        AABBBDD
        AACCCEE
    """
)
f._save_name = f"ROI_crossing_stats_{ROI}"


# split vars
acc = th_crossess.loc[th_crossess.variable == "acceleration"]
ang_acc = th_crossess.loc[th_crossess.variable == "angular_acceleration"]

# get distance (in space) between acc crossess and ang acc crossess
dist = Vector(
    acc.x.values - ang_acc.x.values, acc.y.values - ang_acc.y.values
).magnitude

# draw histogram of time interval
draw.Hist(acc.time, bins=15, alpha=0.4, color=blue_dark, ax=axes["B"])
_ = draw.Hist(ang_acc.time, bins=15, alpha=0.4, color=pink, ax=axes["B"])

# draw histogram of distance
draw.Hist(dist, bins=25, label=r"dist", alpha=0.4, ax=axes["C"])

# draw th crossess over ROI as 2D KDE
draw.ROI(
    ROI, ax=axes["A"],
)
sns.kdeplot(
    data=acc,
    x="x",
    y="y",
    color=blue_dark,
    fill=True,
    levels=6,
    ax=axes["A"],
    alpha=0.5,
)
sns.kdeplot(
    data=ang_acc,
    x="x",
    y="y",
    color=pink,
    fill=True,
    levels=6,
    ax=axes["A"],
    alpha=0.5,
)

sns.kdeplot(
    data=acc, x="x", y="y", color=blue_dark, levels=6, ax=axes["A"], alpha=1,
)
sns.kdeplot(
    data=ang_acc, x="x", y="y", color=pink, levels=6, ax=axes["A"], alpha=1,
)


# get initial speed/acc and speed at turn peak
s_init, s_at_start_of_turn, initial_acc = [], [], []
for n, cross in enumerate(crosses):
    at_turn = np.argmax(cross.thetadot[: int(len(cross) / 3)])

    s_at_start_of_turn.append(cross.speed[at_turn])
    s_init.append(np.mean(cross.speed[:5]))
    initial_acc.append(np.mean(cross.acceleration[:at_turn]))

    # axes['A'].scatter(cross.x[at_turn], cross.y[at_turn], color='k', alpha=.4)

# draw histograms of speed at start and at turn
h1 = draw.Hist(
    s_at_start_of_turn, ax=axes["E"], color="k", label=r"$\vec{v}_{turn}$"
)
draw.Hist(
    s_init, ax=axes["E"], bins=h1.bins, color="g", label=r"$\vec{v}_{init}$"
)

# ddraw linear regression on initial speed vs mean acceleration
data = pd.DataFrame(dict(initial_speed=s_init, mean_acceleration=initial_acc))
_ = sns.regplot(
    x="initial_speed",
    y="mean_acceleration",
    data=data,
    scatter_kws=dict(s=5, color=blue_grey_dark, alpha=1),
    line_kws=dict(lw=4, color=pink, zorder=100),
    ax=axes["D"],
)
axes["D"].axhline(0, ls="--", lw=2, color=[0.3, 0.3, 0.3], zorder=-1)


# _ =axes['A'].legend()
# _ =axes['B'].legend()
# _ =axes['E'].legend()

_ = axes["B"].set(xlabel="time to threshold cross")
# _ = axes['C'].set(xlabel=r'|pos_{\vec{a}} - pos_{\ddot \theta}|')
# _ = axes['E'].set(xlabel=r'$speed \frac{cm}{s}')


recorder.add_figures()
# %%
# plot ROI
f, ax = plt.subplots(figsize=(8, 12))
f._save_name = ROI
draw.Hairpin()

draw.ROI(ROI, shade=True)

recorder.add_figures()


# %%
ttest_ind(
    th_crossess.loc[th_crossess.variable == "acceleration"].time,
    th_crossess.loc[th_crossess.variable == "angular_acceleration"].time,
)
