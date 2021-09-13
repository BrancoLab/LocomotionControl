# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")

from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

from fcutils.plot.elements import plot_mean_and_error

from data import colors
from data.dbase import db_tables
from data import data_utils


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

    for variable in ("speed", "angular_velocity"):
        aligned = {n: [] for n in bins}

        for i, bout in bouts.iterrows():
            variable_data = tracking[bout["name"]][variable][
                bout.start_frame : bout.end_frame
            ]

            variable_data = data_utils.convolve_with_gaussian(
                variable_data, 11
            )
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

        data[variable] = np.array([np.nanmean(v) for v in aligned.values()])
        data[variable + "_std"] = np.array(
            [np.nanstd(v) for v in aligned.values()]
        )

    data["global_coord"] = bins
    return pd.DataFrame(data).interpolate(axis=1)


# get bouts
bouts = pd.DataFrame(
    (
        db_tables.LocomotionBouts & 'complete="true"' & 'direction="outbound"'
    ).fetch()
)
logger.info(f"Found {len(bouts)} complete bouts")

# get tracking for each session
sessions = bouts["name"].unique()
logger.debug(f"Getting tracking for {len(sessions)} sessions")
tracking = {}
for sess in sessions:
    tracking[sess] = db_tables.Tracking.get_session_tracking(
        sess, body_only=True
    )

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
    x = trk["x"][bout.start_frame : bout.end_frame]
    y = trk["y"][bout.start_frame : bout.end_frame]
    speed = trk["speed"][bout.start_frame : bout.end_frame]
    avel = trk["angular_velocity"][bout.start_frame : bout.end_frame]

    speed = data_utils.convolve_with_gaussian(speed, 11)
    avel = data_utils.convolve_with_gaussian(avel, 11)

    coord = trk["global_coord"][bout.start_frame : bout.end_frame]

    # plot tracking
    axes["A"].scatter(x, y, c=coord, vmin=0, vmax=1, cmap="bwr")

    # plot speed and ang vel
    time = np.linspace(0, 1, len(speed))
    axes["B"].plot(coord, speed, color=colors.speed, alpha=0.1)
    axes["C"].plot(coord, avel, color=colors.angular_velocity, alpha=0.1)

axes["C"].axhline(0, lw=2, ls="--", color=[0.3, 0.3, 0.3], zorder=-1)

# plot mean and std of velocities
plot_mean_and_error(
    aligned_tracking.speed,
    aligned_tracking.speed_std,
    axes["B"],
    x=aligned_tracking.global_coord,
    color="salmon",
)
plot_mean_and_error(
    aligned_tracking.angular_velocity,
    aligned_tracking.angular_velocity_std,
    axes["C"],
    x=aligned_tracking.global_coord,
    color="salmon",
)

# histogram of bouts durations
axes["D"].hist(bouts.duration, bins=20, color=[0.3, 0.3, 0.3])


axes["C"].set(ylim=[-500, 500])

# %%


# %%
