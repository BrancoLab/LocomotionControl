import sys
from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
from random import choice


from fcutils.path import files, from_json
from fcutils.progress import track
from fcutils.maths.geometry import (
    calc_distance_between_points_in_a_vector_2d as get_speed_from_xy,
)
from fcutils.maths.coordinates import R, M
from fcutils.maths import rolling_mean

sys.path.append("./")
from experimental_validation._tracking import clean_dlc_tracking

from data_wrangling import paths


"""
    Load and clean DLC output
    convert units to cm and seconds
    check if mice stop during trial
    discard trials too short

    save results in a single trials dataframe
"""

START_SPEED_TH = 10  # cm/s
SPEED_TH = 4  # cm/s
DURATION_TH = 0.4  # s
DISTANCE_TH = 15  # cm
MIRROR_AXES = ("x", "y", "origin", "xy")


# load videos metadata
videos_matadata = from_json(paths.videos_metadata_path)

save_path = paths.main_folder / "TRIALS.h5"
trials = dict(id=[], x=[], y=[],)

f, ax = plt.subplots(figsize=(8, 8))
ax.set(xlabel="X cm", ylabel="Y cm")
ax.axis("equal")
ax.scatter(0, 0, s=100, color="salmon", zorder=-1)


def append(name, xy, show=True):
    trials["id"].append(name)
    trials["x"].append(xy[:, 0])
    trials["y"].append(xy[:, 1])

    if show:
        ax.plot(xy[:, 0], xy[:, 1], color="k", alpha=0.4)


# go over each trial
tfiles = files(paths.tracking_folder, "*.h5")
durations, distances = [], []
original = 0
for num, tfile in track(enumerate(tfiles[::-1]), total=len(tfiles)):
    TNAME = tfile.stem.split("DLC")[0]

    # get metadata
    try:
        metadata = videos_matadata[TNAME]
    except KeyError:
        logger.info(f"Could not find metadata for {TNAME}")
        continue
    continue

    # get trackig data
    try:
        tracking = clean_dlc_tracking(pd.read_hdf(tfile))[0]["body"]
    except ValueError:
        continue

    # interpolate where likelihood is low
    like = tracking["likelihood"].values
    tracking.drop("likelihood", axis=1)
    tracking[like < 0.999] = np.nan
    tracking = tracking.interpolate(axis=0)

    # dataframe -> np array
    xy = tracking[["x", "y"]].values

    # convert from pixels to cms
    xy *= metadata["cm_per_px"]

    # smooth data
    xy = np.vstack(
        [
            rolling_mean(xy[:, 0], int(metadata["fps"] / 4)),
            rolling_mean(xy[:, 1], int(metadata["fps"] / 4)),
        ]
    ).T

    # cut trial to start
    xy = xy[metadata["nframes_before"] + int(0.25 * metadata["fps"]) :, :]

    # get speed
    speed = get_speed_from_xy(xy[:, 0], xy[:, 1]) * metadata["fps"]
    speed[0] = speed[1]
    if np.any(speed > 120):
        logger.warning(f"Detected very high speed in trial: {TNAME}")
        continue

    # get start to when speed is high enough/
    try:
        start = np.where(speed > START_SPEED_TH)[0][0]
    except IndexError:
        logger.warning(f"{TNAME} speed never crossed speed thershold")
        continue
    xy = xy[start:]

    # cut to when speed goes below threshold
    try:
        stop = np.where(speed[start:] < SPEED_TH)[0][0]
    except IndexError:
        logger.debug("Did get slow speed")
        stop = -1
    xy = xy[:stop, :]

    # check if trial is too short
    duration = len(xy) / metadata["fps"]
    if duration < DURATION_TH:
        logger.warning(f"Trial {TNAME} is too brief: {round(duration, 3)}")
        continue

    durations.append(duration)

    # check distance travelled
    dist = np.sum(np.abs(speed[:stop])) / metadata["fps"]
    if dist < DISTANCE_TH:
        logger.warning(
            f"Trial {TNAME} is too short distance: {round(dist, 3)}"
        )
        continue
    distances.append(dist)

    # center at origin
    xy -= xy[0, :]

    # append original trial
    append(tfile.stem, xy)
    original += 1

    # augment data rotation and mirror
    for i in range(5):
        angle = npr.uniform(-360, 360)
        _xy = (R(angle) @ xy.T).T
        append(tfile.stem + f"_rot_{i}", _xy, show=False)
        append(
            tfile.stem + f"_mir_{i}",
            (M(choice(MIRROR_AXES)) @ _xy.T).T,
            show=False,
        )

    # if num > 105:
    #     break

# plot stuff
f, axs = plt.subplots(ncols=2, figsize=(12, 8), sharey=True)
axs[0].hist(durations)
axs[1].hist(distances)
axs[0].set(title="Durations (s)")
axs[1].set(title="Distances (cm)")
plt.show()


# save data
trials = pd.DataFrame(trials)
logger.info(
    f"Saving {len(trials)} augmented trials to file [{original} original trials]\n\n\n\n"
)
trials.to_hdf(save_path, key="hdf")
