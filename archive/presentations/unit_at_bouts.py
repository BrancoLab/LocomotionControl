# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from tpd import recorder

from data import paths
from data.data_structures import LocomotionBout, merge_locomotion_bouts
from data.dbase import db_tables

import draw
from kinematics import time
from matplotlib import rc


# %%
# --------------------------------- get ephys -------------------------------- #
session = "FC_210721_AAA1110750_hairpin"
logger.info("fetching units")
recording = (db_tables.Recording & f'name="{session}"').fetch(as_dict=True)[0]
cf = recording["recording_probe_configuration"]
logger.info("Fetching ephys data")
units = db_tables.Unit.get_session_units(
    session, cf, spikes=True, firing_rate=True, frate_window=100,
)
units["probe_configuration"] = [cf] * len(units)
rsites = pd.DataFrame(
    (
        db_tables.Probe.RecordingSite
        & recording
        & f'probe_configuration="{cf}"'
    ).fetch()
)
logger.info(f"Found {len(units)} units")
# %%
# %%
# --------------------------------- get bouts -------------------------------- #
# load all bouts
_bouts = pd.read_hdf(
    paths.analysis_folder / "behavior" / "saved_data" / f"complete_bouts.h5"
).sort_values("duration")


sessions = (
    db_tables.ValidatedSession * db_tables.Session & "is_recording=1"
).fetch("name")

sessions_with_bouts = [s for s in sessions if s in _bouts["name"].unique()]
print(sessions_with_bouts)
# %%


_bouts = _bouts.loc[(_bouts.duration < 20) & (_bouts["name"] == session)]
# _bouts = _bouts.iloc[:200]
print(f"Kept {len(_bouts)} bouts")

bouts = []
for i, bout in _bouts.iterrows():
    bouts.append(LocomotionBout(bout))

# merge bouts for heatmaps
X, Y, S, A, T, AV, AA = merge_locomotion_bouts(bouts)
