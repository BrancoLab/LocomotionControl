# %%
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path
import pandas as pd
import sys
import os

from fcutils.progress import track
from fcutils.maths.signals import rolling_mean

# from fcutils.maths.coordinates import R
# from fcutils.maths.geometry import (
#     calc_angle_between_vectors_of_points_2d as get_orientation,
# )


module_path = Path(os.path.abspath(os.path.join("."))).parent
sys.path.append(str(module_path))
sys.path.append("./")
from experimental_validation.trials import Trials
from control import paths


# ? This is to clean up/augment real tracking  data before running control on it
# ! this is NOT to clean up the data produced by CONTROL before running RNN

"""
    Take raw tracking from the psychometric mazes and clean it up 
    to prepare a dataset for control.

    Steps:
        1) load trials tracking from folder
        2) augment data by mirroring
        3) save and generate bash scripts

    # TODO NOW ======
        - re-compute omega and theta (get bone angle...)
        - generate bash files

    # TODO for the future
        - make this work for any experiment type
        - other augment types (e.g. augment by rotation with R)
        - more quality controls
"""

# %%

# --------------------------------- Get data --------------------------------- #


X_mirror, Y_mirror = 25, 25  # coordinates around which to mirror data

trials = Trials(only_tracked=True)

# collate all trials and plot them
f, ax = plt.subplots(figsize=(10, 10))

collated = dict(ID=[], x=[], y=[], theta=[], v=[], omega=[])
n = 0
for trial in track(trials, description="Collating", transient=True):
    # smooth variables
    x = rolling_mean(trial.body.x, 6)
    y = rolling_mean(trial.body.y, 6)
    theta = rolling_mean(trial.orientation, 6)
    v = rolling_mean(trial.v, 6)
    omega = rolling_mean(trial.omega, 6)

    # add and augment
    # for mirror in (None, 'x', 'y'):
    #     if mirror == 'x':
    #         _x = X_mirror - x.copy() + X_mirror
    #         _y = y.copy()
    #     elif mirror == 'y':
    #         _y = Y_mirror - y.copy() + Y_mirror
    #         _x = x.copy()
    #     else:
    #         _x, _y = x.copy(), y.copy()

    collated["ID"].append(n)
    n += 1

    collated["x"].append(x)
    collated["y"].append(y)
    collated["theta"].append(theta)
    collated["v"].append(v)
    collated["omega"].append(omega)

    ax.plot(x, y)

trials = pd.DataFrame(collated)
logger.info(f"After augmentation we have: {len(trials)} trials")

# save
trials.to_hdf(paths.trials_cache, key="hdf")


# %%

# ------------------------------- generate bash ------------------------------ #

template = """#! /bin/bash

#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 1gb # memory pool for all cores
#SBATCH --job-name="tr NNN"
#SBATCH -n 1
#SBATCH --time=02:00:00
#SBATCH	-o logs/tr_NNN.out
#SBATCH -e logs/tr_NNN.err


echo "loading conda env"
module load miniconda

conda activate locomotion

echo "Updating locomotion repobash  "
cd LocomotionControl

echo "locomoting"
python winstor.py  --trialn NNN


"""
save_fld = Path(
    "Z:\\swc\\branco\\Federico\\Locomotion\\control\\trials_bash_files"
)

for n, (i, t) in track(enumerate(trials.iterrows()), total=len(trials)):
    txt = template.replace("NNN", str(n))

    with open(save_fld / f"trial_{n}.sh", "w") as out:
        out.write(txt)
