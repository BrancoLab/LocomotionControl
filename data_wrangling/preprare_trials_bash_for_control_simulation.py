from rich import print
import pandas as pd
import sys

sys.path.append("./")
from control import paths

# load trials
# save_path = paths.main_folder / "TRIALS.h5"
trials = pd.read_hdf(paths.trials_cache, key="hdf")
print(f"[salmon]Loaded {len(trials)} trials from : '{paths.trials_cache}'")
print(
    f"[salmon]To run on HPC: --[b] bash launch_all_trials.sh {len(trials)} [/b]--"
)
