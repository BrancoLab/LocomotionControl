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
from fcutils.maths.geometry import (
    calc_angle_between_points_of_vector_2d as get_dir_of_mvmt_from_xy,
)
from fcutils.maths.coordinates import R, M
from fcutils.maths import rolling_mean, derivative

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

START_SPEED_TH = 6  # cm/s
END_SPEED_TH = 4  # cm/s
DURATION_TH = 0.3  # s
DMOV_TH = 1000
DISTANCE_TH = 10  # cm
MIRROR_AXES = ("x", "y", "origin", "xy")

N_AUGMENTED = 20


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


def plot_bad(speed, start=0, end=None):
    f, ax = plt.subplots()
    ax.plot(speed, color="k", alpha=0.5)
    ax.scatter(start, speed[start], s=100, color="r", zorder=100)
    if end is not None:
        ax.scatter(end, speed[end], s=100, color="k", zorder=100)
    ax.axhline(START_SPEED_TH, color="salmon", label="start  TH")
    ax.axhline(END_SPEED_TH, color="g", label="end  TH")


def get_start(speed, fps):
    maxidx = np.argmax(speed)
    try:

        th_cross = np.where(speed[:maxidx] < START_SPEED_TH)[0][-1]
    except IndexError:
        th_cross = 0
    return th_cross + 1 + int(fps / 5)


# go over each trial
tfiles = files(paths.tracking_folder, "*.h5")  #
logger.info(f"Found {len(tfiles)} tracking files")

durations, distances, dmovs = [], [], []
original = 0
for num, tfile in track(enumerate(tfiles[::-1]), total=len(tfiles)):
    TNAME = tfile.stem.split("DLC")[0]

    # get metadata
    try:
        metadata = videos_matadata[TNAME]
    except KeyError:
        logger.info(f"Could not find metadata for {TNAME}")
        continue

    # get trackig data
    try:
        alltracking = clean_dlc_tracking(pd.read_hdf(tfile))[0]
    except ValueError:
        logger.info(f"Could not find tracking data for {TNAME}")
        continue

    body = alltracking["body"]
    tail = alltracking["tail_base"]

    # interpolate where likelihood is low
    like = body["likelihood"].values
    body.drop("likelihood", axis=1)
    body[like < 0.99999] = np.nan
    body = body.interpolate(axis=0)

    like = tail["likelihood"].values
    tail.drop("likelihood", axis=1)
    tail[like < 0.99999] = np.nan
    tail = tail.interpolate(axis=0)

    # dataframe -> np array
    x = np.vstack([body["x"], tail["x"]]).mean(0)
    y = np.vstack([body["y"], tail["y"]]).mean(0)
    xy = np.vstack([x, y]).T

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
    xy = xy[metadata["nframes_before"] :, :]

    # get speed
    speed = get_speed_from_xy(xy[:, 0], xy[:, 1]) * metadata["fps"]
    speed[0] = speed[1]
    smooth_speed = rolling_mean(speed, 6)
    if np.any(smooth_speed > 140):
        logger.warning(f"Detected very high smooth_speed in trial: {TNAME}")
        continue
    if not np.any(smooth_speed > START_SPEED_TH):
        logger.warning(f"{TNAME}: Speed never crossed start threshold")
        continue

    # get start to when smooth_speed is high enough/
    start = get_start(smooth_speed, metadata["fps"])
    xy = xy[start:]

    # cut to when smooth_speed goes below threshold
    try:
        stop = np.where(smooth_speed[start:] < END_SPEED_TH)[0][0]
    except IndexError:
        logger.debug("Didn't get slow smooth_speed")
        stop = -1
    xy = xy[:stop, :]

    # check if trial is too short
    duration = len(xy) / metadata["fps"]
    if duration < DURATION_TH:
        logger.warning(f"Trial {TNAME} is too brief: {round(duration, 3)}")
        # plot_bad(smooth_speed, start=start, end=stop+start)
        # plt.show()
        continue

    # check distance travelled
    dist = np.sqrt((xy[-1, 0] - xy[0, 0]) ** 2 + (xy[-1, 1] - xy[0, 1]) ** 2)
    if dist < DISTANCE_TH:
        logger.warning(
            f"Trial {TNAME} is too short distance: {round(dist, 3)}"
        )
        continue

    # check if dir of mvmt goes through too many changes
    dmov = get_dir_of_mvmt_from_xy(xy[:, 0], xy[:, 1])
    dmov = np.sum(np.abs(np.degrees(derivative(np.unwrap(np.radians(dmov))))))
    if dmov > DMOV_TH:
        logger.warning(
            f"Trial {TNAME} change in heading direction is too large: {dmov:.2f}"
        )
        continue

    # center at origin
    xy -= xy[0, :]

    # append original trial
    append(tfile.stem + f"_orig", xy)
    durations.append(duration)
    distances.append(dist)
    dmovs.append(dmov)
    original += 1

    # augment data rotation and mirror
    for i in range(N_AUGMENTED):
        angle = npr.uniform(-360, 360)
        _xy = (R(angle) @ xy.T).T
        append(tfile.stem + f"_rot_{i}", _xy, show=False)
        append(
            tfile.stem + f"_mir_{i}",
            (M(choice(MIRROR_AXES)) @ _xy.T).T,
            show=False,
        )

    # if num > 1000:
    #     break


# plot stuff
f, axs = plt.subplots(ncols=3, figsize=(12, 8), sharey=True)
axs[0].hist(durations)
axs[1].hist(distances)
axs[2].hist(dmovs)
axs[0].set(title="Durations (s)")
axs[1].set(title="Distances (cm)")
axs[2].set(title="TOT changes in heading dir (deg)")
plt.show()


# save data
trials = pd.DataFrame(trials)
logger.info(f"Kept {original}/{num} trials, the other were baddd")
logger.info(
    f"Saving {len(trials)} augmented trials to file [{original} original trials]\n\n\n\n"
)
trials.to_hdf(save_path, key="hdf")
