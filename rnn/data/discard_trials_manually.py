# %%
from pigeon import annotate
from IPython.display import display, Image
from pathlib import Path
import sys
from loguru import logger

from fcutils import path

sys.path.append("./")
from control.history import load_results_from_folder

"""
    Take control simulations and select only 'good' ones:
        - the simulation must have covered most of the goal trajectory
        - there shouldn't have been catastrophic failures
        - the simulation's trajectory should be close to goal trajectory for all variables.

    Note: this is not to manually curate trials from behavioural data before running the
    simulations on them, this is to curate the results of the simulations before
    creating datasets for RNN training.

    Curation steps:
        - algorithmically remove catastrophic failures
        - manual pass on data to remove bad ones
        - second manual pass to mae sure everthing's OK
"""


fld = Path("D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\RNN\\training_data")
flds = path.subdirs(fld)

# %%

# ---------------------------------------------------------------------------- #
#                              1. DISCARD FAILURES                             #
# ---------------------------------------------------------------------------- #
logger.info("Step 1: remove catastrophic failures")
to_discard = []
for fld in flds:
    # get the proprtion of trajectory waypoints visited
    history, info, trajectory, _ = load_results_from_folder(fld)

    prop_visited = len(history.trajectory_idx.unique()) / trajectory.shape[0]

    if prop_visited < 0.7:
        to_discard.append(fld)

logger.info(f"Found {len(to_discard)}/{len(flds)} simulations to discard.")
# for fld in to_discard:
#     path.delete(fld)
# flds = path.subdirs(fld)


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


kept = {fld: img for fld, img in images.items() if annotations[img] == "good"}
logger.info(f"Kept {len(kept)}/{len(images)} simulations")

# %%

# ---------------------------------------------------------------------------- #
#                                 3. INSPECTION                                #
# ---------------------------------------------------------------------------- #

# second round of quality checks
annotations = annotate(
    kept.values(),
    options=["good", "bad"],
    display_fn=lambda filename: display(Image(filename)),
)

kept = {fld: img for fld, img in images.items() if annotations[img] == "good"}
logger.info(f"Kept {len(kept)}/{len(images)} simulations")

# %%

# ---------------------------------------------------------------------------- #
#                                  4. CLEANUP                                  #
# ---------------------------------------------------------------------------- #
# remove bad data from folder
for fld in images.keys():
    if fld not in kept.keys():
        path.delete(fld)

flds = path.subdirs(fld)
logger.info(f"At the end {len(flds)} simulations are left.s")
