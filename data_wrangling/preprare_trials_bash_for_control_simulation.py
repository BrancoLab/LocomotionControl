from loguru import logger
from pathlib import Path
import pandas as pd

from fcutils.progress import track

TEST_MODE = True  # only few trials for simplicity

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

# load trials
# save_path = paths.main_folder / "TRIALS.h5"
save_path = "Z:\\swc\\branco\\Federico\\Locomotion\\control\\trials_cache.h5"
trials = pd.read_hdf(save_path, key="hdf")
logger.info(f"Loaded {len(trials)} trials")

if TEST_MODE:
    # select a few random trials
    trials = trials.sample(50)
    logger.info(f"Test mode, kept 50 trials")


for n, (i, t) in track(enumerate(trials.iterrows()), total=len(trials)):
    txt = template.replace("NNN", str(n))

    with open(save_fld / f"trial_{n}.sh", "w") as out:
        out.write(txt)
