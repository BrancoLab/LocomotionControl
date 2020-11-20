from pathlib import Path
import pandas as pd

import sys

sys.path.append("./")
from proj.paths import trials_cache

# import shutil
from rich.progress import track

"""
    Generate a bash script for each trial in the data set to run
    the control on all trials
"""


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
# shutil.rmtree(str(save_fld))
save_fld.mkdir(exist_ok=True)

trials = pd.read_hdf(trials_cache, key="hdf",)
print(trials_cache)
print(f"Loaded {len(trials)} trials")


for n, (i, t) in track(enumerate(trials.iterrows()), total=len(trials)):
    txt = template.replace("NNN", str(n))

    with open(save_fld / f"trial_{n}.sh", "w") as out:
        out.write(txt)
