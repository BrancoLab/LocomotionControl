from rich import print
import pandas as pd
import sys
import numpy as np

sys.path.append("./")
from control import paths

"""
to run simulations on all trials use bash launch_all_trials.sh
this tells you how many trials are ready in the cache for simulations
"""


# load trials
# save_path = paths.main_folder / "TRIALS.h5"
trials = pd.read_hdf(paths.trials_cache, key="hdf")
print(f"[salmon]Loaded {len(trials)} trials from : '{paths.trials_cache}'")
print(
    f"[salmon]To run on HPC: --[b] bash launch_all_trials.sh {int(np.floor(len(trials)/50))} [/b]--which will create N jobs each running 500 trials"
)
