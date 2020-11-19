# %%
from pathlib import Path
import pandas as pd
from proj.paths import trials_cache
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
#SBATCH --time=00:30:00
#SBATCH	-o out.out
#SBATCH -e err.err


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

# %%
trials = pd.read_hdf(trials_cache)
print(f"Loaded {len(trials)} trials")


# %%
for i, t in track(trials.iterrows(), total=len(trials)):
    txt = template.replace("NNN", str(i))

    with open(save_fld / f"trial_{i}.sh", "w") as out:
        out.write(txt)
# %%
