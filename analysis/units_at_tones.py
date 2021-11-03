# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tpd import recorder

from data import paths
from data.data_structures import LocomotionBout
import draw
from data.dbase import db_tables
from loguru import logger


'''
    It plots rasters with spikes of each unit aligned to tone onsets and offsets for each session
'''



# %%
# ------------------------------- load tracking ------------------------------ #
MIN_DUR = 6.5

_bouts = []
for roi in ('T3',): #, 'T2', 'T3', 'T4'):
    _bouts.append(pd.read_hdf(
        paths.analysis_folder / "behavior" / "saved_data" / f"{roi}_crossings.h5"
    ))
_bouts = pd.concat(_bouts).sort_values("duration")
_bouts = _bouts.loc[_bouts.duration < MIN_DUR]


# get sessions and tracking
sessions = (db_tables.ValidatedSession * db_tables.Session & 'is_recording=1').fetch('name')
logger.info(f'Found {len(sessions)} recordings')
print(sessions, sep='\n')



# %%

# select a session
session = 'FC_210721_AAA1110750_hairpin'

# ----------------------------- load probe/units ----------------------------- #
logger.info('fetching units')
recording = (db_tables.Recording & f'name="{session}"').fetch(as_dict=True)[0]
cf = recording['recording_probe_configuration']
logger.info("Fetching ephys data")
units = db_tables.Unit.get_session_units(
    session,
    cf,
    spikes=True,
    firing_rate=True,
    frate_window=100,
)
units['probe_configuration'] = [cf] * len(units)
rsites = pd.DataFrame((db_tables.Probe.RecordingSite & recording & f'probe_configuration="{cf}"').fetch())
logger.info(f'Found {len(units)} units')


# %%
# ? PLOT rasters for each UNIT
regions = [x for x in sorted(units.brain_region.unique())]
regions = [x for x in regions if x in ('CUN', 'PPN', 'ICe')]
_units = {r:units.loc[units.brain_region == r].reset_index() for r in regions}
n_units = max([len(x) for x in _units.values()])

# get tones on/off
tones = db_tables.Tones.get_session_tone_on(session)
tones_on = np.where(np.diff(tones) == 1)[0]
tones_off = np.where(np.diff(tones) == -1)[0]
starts_on = [t-60 for t in tones_on]
ends_on = [t+60 for t in tones_on]

starts_off = [t-60 for t in tones_off]
ends_off = [t+60 for t in tones_off]

f, axes = plt.subplots(figsize=(20, 20), nrows=n_units, ncols=len(regions)*2, sharex=True, sharey=True)
# f.suptitle(ROI)

for rn, (region, uns) in enumerate(_units.items()):
    ax_num = rn *2
    axes[0, ax_num].set(title=region)
    axes[0, ax_num+1].set(title='tone off')

    for n, unit in uns.iterrows():
        axes[n, ax_num].axvline(60, lw=2, color='k')
        draw.Raster(unit.spikes, starts_on, ends_on, ax=axes[n, ax_num], color=unit.color, lw=2)
        axes[n, ax_num].set(ylabel=unit.unit_id)

        axes[n, ax_num+1].axvline(60, lw=2, color='k')
        draw.Raster(unit.spikes, starts_off, ends_off, ax=axes[n, ax_num+1], color=unit.color, lw=2)
        axes[n, ax_num+1].set(ylabel=unit.unit_id)

# %%
