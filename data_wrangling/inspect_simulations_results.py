# %%
from pigeon import annotate
from IPython.display import display, Image
from pathlib import Path
from loguru import logger
import sys
import os

module_path = Path(os.path.abspath(os.path.join("."))).parent.parent

from fcutils import path
from fcutils.progress import track

sys.path.append(str(module_path))
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


fld = Path(
    "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\RNN\\training_data_temp"
)
flds = path.subdirs(fld)
if flds is None:
    raise ValueError("Did not find any simulations folders")
logger.info(f"Found {len(flds)} folders")

# %%

# ---------------------------------------------------------------------------- #
#                              1. DISCARD FAILURES                             #
# ---------------------------------------------------------------------------- #
logger.info("Step 1: remove catastrophic failures")
to_discard = []
for subfld in track(flds):
    # get the proprtion of trajectory waypoints visited
    try:
        history, info, trajectory, _ = load_results_from_folder(subfld)
    except FileNotFoundError:
        logger.debug(f"Skipped {subfld}")
        to_discard.append(subfld)
        continue

    prop_visited = len(history.trajectory_idx.unique()) / trajectory.shape[0]

    if prop_visited < 0.8:
        to_discard.append(subfld)

logger.info(f"Found {len(to_discard)}/{len(flds)} simulations to discard.")
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

# get paths to all images
images = {
    fld: path.files(fld, "outcome.png")
    for fld in flds
    if path.files(fld, "outcome.png")
}
lookup = {v: k for k, v in images.items()}


# do a first round of manual selection
annotations = annotate(
    images.values(),
    options=["good", "bad"],
    display_fn=lambda filename: display(Image(filename)),
)

# %%
annotations_dict = {k: v for (k, v) in annotations}
_images = {
    fld: img for fld, img in images.items() if img in annotations_dict.keys()
}
discard = {
    fld: img for fld, img in _images.items() if annotations_dict[img] == "bad"
}
logger.info(f"Discarded {len(discard)}/{len(images)} simulations")

for subfld in discard.keys():
    path.delete(subfld)
flds = path.subdirs(fld)
logger.info(f"We are left with {len(flds)} folders")

# %%
