# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")

from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import time

from fcutils.plot.elements import plot_mean_and_error
from fcutils.progress import track

from data import colors
from data.dbase import db_tables
from data import data_utils


# get bouts
bouts = pd.DataFrame(
    (
        db_tables.LocomotionBouts
        & 'complete="true"'
        & 'direction="outbound"'
        & "duration<8"
    ).fetch()
)
logger.info(f"Found {len(bouts)} complete bouts")

# get tracking for each session
sessions = bouts["name"].unique()
logger.debug(f"Getting tracking for {len(sessions)} sessions")
tracking = {}
for sess in track(sessions):
    tracking[sess] = db_tables.Tracking.get_session_tracking(
        sess, body_only=True
    )
    time.sleep(1)

#  %%


def align_in_space(
    bouts: pd.DataFrame, tracking: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
        Get the average and standard deviation of speed and angular velocity
        as a function of position in the hairpin track
    """
    data = dict(
        speed=[], speed_std=[], angular_velocity_std=[], angular_velocity=[]
    )
    bins = np.linspace(0, 1, 51)
    delta = np.mean(np.diff(bins))

    for variable in ("speed", "angular_velocity", "dmov_velocity"):
        logger.info(f"Aligning {variable}")
        aligned = {n: [] for n in bins}

        for i, bout in bouts.iterrows():
            variable_data = tracking[bout["name"]][variable][
                bout.start_frame : bout.end_frame
            ]

            try:
                variable_data = data_utils.convolve_with_gaussian(
                    variable_data, 11
                )
            except ValueError:
                raise ValueError(bout)
            coord = tracking[bout["name"]]["global_coord"][
                bout.start_frame : bout.end_frame
            ]

            # get data to aligned in space
            for n, th in enumerate(bins):
                if n == 0:
                    pre, post = 0, delta
                else:
                    pre, post = th - delta, th + delta
                aligned[th].extend(
                    variable_data[(coord > pre) & (coord <= post)]
                )

        data[variable] = np.array([np.median(v) for v in aligned.values()])
        data[variable + "_std"] = np.array(
            [np.nanstd(v) for v in aligned.values()]
        )

    data["global_coord"] = bins
    return pd.DataFrame(data).interpolate(axis=1)


# get tracking data aligned to space
aligned_tracking = align_in_space(bouts, tracking)

# %%
# make figure


f = plt.figure(figsize=(19, 8))
axes = f.subplot_mosaic(
    """
    ABBD
    ACCE
    """
)
f._save_name = "complete_bouts"
f.suptitle("complete bouts")

aligned = {n: [] for n in np.linspace(0, 1, 101)}
for i, bout in bouts.iterrows():
    trk = tracking[bout["name"]]
    x = trk["x"][bout.start_frame : bout.end_frame].copy()
    y = trk["y"][bout.start_frame : bout.end_frame].copy()
    speed = trk["speed"][bout.start_frame : bout.end_frame].copy()
    avel = trk["dmov_velocity"][bout.start_frame : bout.end_frame].copy()
    dmov = trk["direction_of_movement"][
        bout.start_frame : bout.end_frame
    ].copy()
    coord = trk["global_coord"][bout.start_frame : bout.end_frame].copy()

    # plot tracking
    # axes["A"].scatter(x, y, c=coord, cmap="tab10", vmin=0, vmax=1, alpha=.5)
    axes["A"].scatter(x, y, c=dmov, cmap="hsv", alpha=0.5)

    # plot speed and ang vel
    time = np.linspace(0, 1, len(speed))
    axes["B"].scatter(coord, speed, color=colors.speed, s=20, alpha=0.1)
    axes["C"].scatter(
        coord, avel, color=colors.angular_velocity, s=20, alpha=0.1
    )

    # plt speed vs ang vel
    axes["E"].scatter(speed, np.abs(avel), color="k", alpha=0.01)

axes["C"].axhline(0, lw=2, ls="--", color=[0.3, 0.3, 0.3], zorder=-1)

# plot mean and std of velocities
plot_mean_and_error(
    aligned_tracking.speed,
    aligned_tracking.speed_std,
    axes["B"],
    x=aligned_tracking.global_coord,
    color="red",
)
plot_mean_and_error(
    aligned_tracking.dmov_velocity,
    aligned_tracking.dmov_velocity_std,
    axes["C"],
    x=aligned_tracking.global_coord,
    color="red",
)

for ax, Y in zip("BC", (0, -600)):
    axes[ax].scatter(
        np.linspace(0, 1, 250),
        np.ones(250) * Y,
        c=np.linspace(0, 1, 250),
        cmap="tab10",
        zorder=100,
    )

# histogram of bouts durations
_ = axes["D"].hist(bouts.duration, bins=20, color=[0.3, 0.3, 0.3])

#
axes["B"].set(xlim=[0.1, 1])
_ = axes["C"].set(xlim=[0.1, 0.75], ylim=[-1000, 1000])

# %%


# %%
trials = dict(x=[], y=[], speed=[], dmov=[], dmov_velocity=[], gcoord=[])

for i, bout in bouts.iterrows():
    trk = tracking[bout["name"]]

    trials["x"].append(trk["x"][bout.start_frame : bout.end_frame].copy())
    trials["y"].append(trk["y"][bout.start_frame : bout.end_frame].copy())
    trials["speed"].append(
        trk["speed"][bout.start_frame : bout.end_frame].copy()
    )
    trials["dmov_velocity"].append(
        trk["dmov_velocity"][bout.start_frame : bout.end_frame].copy()
    )
    trials["dmov"].append(
        trk["direction_of_movement"][bout.start_frame : bout.end_frame].copy()
    )
    trials["gcoord"].append(
        trk["global_coord"][bout.start_frame : bout.end_frame].copy()
    )

trials = pd.DataFrame(trials)
trials.to_hdf(
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\rnn_simulations\bouts.h5",
    key="hdf",
)
# %%
