import sys
from loguru import logger
import pandas as pd
import numpy as np
import numpy.random as npr


from fcutils.path import files, from_json
from fcutils.progress import track
from fcutils.maths.geometry import (
    calc_distance_between_points_in_a_vector_2d as get_speed_from_xy,
)
from fcutils.maths.coordinates import R, M

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

SPEED_TH = 1  # cm/s
DURATION_TH = 2  # s


# load videos metadata
videos_matadata = from_json(paths.videos_metadata_path)

save_path = paths.main_folder / "TRIALS.h5"
trials = dict(id=[], x=[], y=[],)


def append(name, xy):
    trials["id"].append(name)
    trials["x"].append(xy[:, 0])
    trials["y"].append(xy[:, 1])


# go over each trial
original = 0
for tfile in track(files(paths.tracking_folder, "*.h5")):
    # get the data
    tracking = clean_dlc_tracking(pd.read_hdf(tfile))[0]["body"]

    # interpolate where likelihood is low
    like = tracking["likelihood"].values
    tracking.drop("likelihood", axis=1)
    tracking[like < 0.999] = np.nan
    tracking = tracking.interpolate(axis=0)

    # dataframe -> np array
    xy = tracking[["x", "y"]].values

    # center at origin
    xy -= xy[0, :]

    # convert from pixels to cms
    metadata = videos_matadata[tfile.stem.split("Deep")[0]]
    xy *= metadata["cm_per_px"]

    # cut trial
    xy = xy[metadata["nframes_before"] + int(0.5 * metadata["fps"]) :, :]

    # get speed
    speed = get_speed_from_xy(xy[:, 0], xy[:, 1]) * metadata["fps"]

    # cut to when speed goes below threshold
    try:
        stop = np.where(speed < SPEED_TH)[0][0]
    except IndexError:
        stop = -1
    tracking = tracking[:-1]

    # check if trial is too short
    duration = len(xy) / metadata["fps"]
    if duration < SPEED_TH:
        logger.warning(
            f"Trial {tfile.stem} is too short: {round(duration, 3)}"
        )
        continue

    # append original trial
    append(tfile.stem, xy)
    original += 1

    # augment data 1: rotation
    for i in range(5):
        angle = npr.uniform(-180, 180)
        append(tfile.stem + f"_rot_{i}", R(angle) @ xy)

    # augment data 2: mirroring
    for i, axis in enumerate(("x", "y", "origin", "xy")):
        append(tfile.stem + f"_mir_{i}", M(axis) @ xy)

# save data
trials = pd.DataFrame(trials)
logger.info(
    f"Saving augmented {len(trials)} to file [{original} original trials]"
)
trials.to_hdf(save_path, key="hdf")
