# %%
# from pigeon import annotate
# from IPython.display import display, Image
from pathlib import Path
from loguru import logger
import numpy as np
import sys
import os

module_path = Path(os.path.abspath(os.path.join("."))).parent

from fcutils import path
from fcutils.progress import track

sys.path.append(str(module_path))
sys.path.append("./")
from control.history import load_results_from_folder

"""
    Take control simulations and select only 'good' ones:
        - the simulation must have covered most of the goal trajectory
        - there shouldn't have been catastrophic failures
        - the simulation's trajectory should be close to goal trajectory for all variables.

    Note: this is **not** to manually curate trials from behavioural data before running the
    simulations on them, this is to curate the results of the simulations before
    creating datasets for RNN training.

    Curation steps:
        - algorithmically remove catastrophic failures
        - manual pass on data to remove bad ones
"""


fld = Path(r"Z:\swc\branco\Federico\Locomotion\control\data - Copy")
flds = path.subdirs(fld)
if flds is None:
    raise ValueError("Did not find any simulations folders")
logger.info(f"Found {len(flds)} folders")

# %%


def get_trajectory_waypoint_at_frame(history, trajectory):
    """
        Get trajectory waypoint at each frame
    """
    return np.vstack([trajectory[w, :] for w in history.trajectory_idx])


def endpoint_distance(history, trajectory):
    """
        Distance between history end point and trajectory end point
    """
    return np.linalg.norm(
        history[["x", "y"]].values[-1, :2] - trajectory[-1, :2]
    )


def max_trajectory_distance(history, trajectory):
    """
        Get max distance from trajectory during simulation
    """
    htraj = get_trajectory_waypoint_at_frame(history, trajectory)
    return np.max(np.linalg.norm(htraj[:, :2] - history[["x", "y"]].values))


# %%
# ---------------------------------------------------------------------------- #
#                              1. DISCARD FAILURES                             #
# ---------------------------------------------------------------------------- #

MAX_DIST = 15  # cm, max distance from trajectory
END_POINT_DIST_TH = 8  # cm, distance between history and traj ends

logger.info("Step 1: remove catastrophic failures")
to_discard = []
for n, subfld in track(enumerate(flds), total=len(flds)):
    # get the proprtion of trajectory waypoints visited
    try:
        history, info, trajectory, _ = load_results_from_folder(subfld)
    except FileNotFoundError:
        logger.debug(f"Skipped {subfld.name}")
        to_discard.append(subfld)
        continue

    if not len(history) or not len(trajectory):
        logger.debug(f"{subfld.name} has empty history")
        to_discard.append(subfld)
        continue

    # check end point distance
    if endpoint_distance(history, trajectory) > END_POINT_DIST_TH:
        logger.debug(f"{subfld.name} end point distance too large")
        to_discard.append(subfld)
        continue

    # check max distance
    if max_trajectory_distance(history, trajectory) > MAX_DIST:
        logger.debug(f"{subfld.name} max distance too large")
        to_discard.append(subfld)
        continue


logger.info(f"Found {len(to_discard)}/{len(flds)} simulations to discard.")


# %%
# delete bad folders
for subfld in to_discard:
    try:
        path.delete(subfld)
    except Exception:
        continue
flds = path.subdirs(fld)


# %%

# ---------------------------------------------------------------------------- #
#                                2. MANUAL PASS                                #
# ---------------------------------------------------------------------------- #

# # get paths to all images
# images = {
#     fld: path.files(fld, "outcome.png")
#     for fld in flds
#     if path.files(fld, "outcome.png")
# }
# lookup = {v: k for k, v in images.items()}


# # do a first round of manual selection
# annotations = annotate(
#     images.values(),
#     options=["good", "bad"],
#     display_fn=lambda filename: display(Image(filename)),
# )

# # %%
# annotations_dict = {k: v for (k, v) in annotations}
# _images = {
#     fld: img for fld, img in images.items() if img in annotations_dict.keys()
# }
# discard = {
#     fld: img for fld, img in _images.items() if annotations_dict[img] == "bad"
# }
# logger.info(f"Discarded {len(discard)}/{len(images)} simulations")

# for subfld in discard.keys():
#     path.delete(subfld)
# flds = path.subdirs(fld)
# logger.info(f"We are left with {len(flds)} folders")

# # %%
