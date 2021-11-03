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

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=False)

'''
    It plots rasters with spikes of each unit at each ROI crossing
    from a selected experiments.
'''



# %%
# ------------------------------- load tracking ------------------------------ #
MIN_DUR = 4

_bouts = []
for roi in ('T2',): #, 'T2', 'T3', 'T4'):
    _bouts.append(pd.read_hdf(
        paths.analysis_folder / "behavior" / "saved_data" / f"{roi}_crossings.h5"
    ))
_bouts = pd.concat(_bouts).sort_values("duration")
_bouts = _bouts.loc[_bouts.duration < MIN_DUR]


# get sessions and tracking
sessions = (db_tables.ValidatedSession * db_tables.Session & 'is_recording=1').fetch('name')
logger.info(f'Found {len(sessions)} recordings')

# filter bouts in recordings
for session in sessions:
    bouts = _bouts.loc[_bouts['name'] == session]
    if not bouts.empty:
        
        print(f'{session} - Found: ', len(bouts), ' ROI crossings')
    else:
        print(f'{session} ---- EMPTY')

# select a session
session = 'FC_210721_AAA1110750_hairpin'
bouts = _bouts.loc[_bouts['name'] == session]
bouts = [LocomotionBout(bout) for i,bout in bouts.iterrows()]

# %%
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
tones = np.where(np.diff(tones) == 1)[0]
starts = [t-60 for t in tones]
ends = [t+60 for t in tones]

f, axes = plt.subplots(figsize=(20, 20), nrows=n_units, ncols=len(regions), sharex=True, sharey=True)
# f.suptitle(ROI)

for rn, (region, uns) in enumerate(_units.items()):
    axes[0, rn].set(title=region)
    for n, unit in uns.iterrows():
        starts = [b.start_frame for b in bouts]
        ends = [b.end_frame for b in bouts]
        # axes[n, rn].axvline(60, lw=2, color='k')

        draw.Raster(unit.spikes, starts, ends, ax=axes[n, rn], color=unit.color, lw=5)
        axes[n, rn].set(ylabel=unit.unit_id)

        # break
    for c in np.arange(n+1, n_units):
        axes[c, rn].axis('off')


# %%
from data.data_utils import convolve_with_gaussian
import seaborn as sns

f = plt.figure(figsize=(20, 12), constrained_layout=True)
f.suptitle(f'{unit.brain_region} - {unit.unit_id}')

axes = f.subplot_mosaic(
    '''
        AABBCCFF
        AADDEEGG
    '''
)


unit = units.loc[units.unit_id == 1036].iloc[0]
frate = convolve_with_gaussian(unit.firing_rate)

draw.ROI(roi, ax=axes['A'])
X, Y = [], []
for bout in bouts:
    draw.Tracking(bout.x, bout.y-2, ax=axes['A'])
    axes['B'].plot(bout.velocity.magnitude, color='k', lw=.5)
    axes['C'].plot(bout.thetadot, color='k', lw=.5)

    axes['D'].plot(frate[bout.start_frame:bout.end_frame])
    X.extend(list(frate[bout.start_frame+2:bout.end_frame-2]))
    Y.extend(list(bout.speed))

sns.regplot(X, Y, ax=axes['E'])

starts = [b.start_frame for b in bouts]
ends = [b.end_frame for b in bouts]

draw.Raster(unit.spikes, starts, ends)




# %%
'''
    Plot firing rate vs speed as mouse goes around a bend
'''
from kinematics import time
from geometry.utils import resample_linear_1d

registered_bouts = time.time_rescale([bout.path.smooth(21) for bout in bouts])


# cut firing rates
frate = convolve_with_gaussian(unit.firing_rate, kernel_width=11)
frates = [frate[locbout.start_frame:locbout.end_frame] for locbout in bouts]

n_frames = np.min([len(fr) for fr in frates])
resampled_frates = [resample_linear_1d(fr, n_frames) for fr in frates]

# %%
f = plt.figure(figsize=(20, 12), constrained_layout=True)
f.suptitle(f'{unit.brain_region} - {unit.unit_id}')

unit = units.loc[units.unit_id == 1036].iloc[0]
frate = convolve_with_gaussian(unit.firing_rate)

f.suptitle(f'{unit.brain_region} - {unit.unit_id}')

axes = f.subplot_mosaic(
    '''
        AABBCC
        AADDEE
    '''
)


draw.ROI(roi, ax=axes['A'])

for bout in bouts:
    draw.Tracking(bout.x, bout.y-2, ax=axes['A'])

    bout_spikes = unit.spikes[(unit.spikes > bout.start_frame)&(unit.spikes < bout.end_frame)] - bout.start_frame
    bout_spikes = bout_spikes[bout_spikes < len(bout.x)]
    draw.Tracking.scatter(bout.x[bout_spikes], bout.y[bout_spikes]-2, ax=axes['A'], color='r', zorder=100, alpha=.2)



for n, bout in enumerate(registered_bouts):
    locbout = bouts[n]

    axes['B'].plot(bout.speed, lw=.5, color='k')
    axes['C'].plot(bout.acceleration_mag, lw=.5, color='k')
    axes['D'].plot(resampled_frates[n], lw=.5, color='k')


axes['B'].plot(np.mean(np.vstack([b.speed for b in registered_bouts]), 0), lw=3, color='r')
axes['C'].plot(np.mean(np.vstack([b.acceleration_mag for b in registered_bouts]), 0), lw=3, color='r')
axes['D'].plot(np.mean(np.vstack(resampled_frates), 0), lw=3, color='r')


# %%
plt.plot(bout.acceleration_mag)
sm = bout.smooth(20)

plt.plot(sm.acceleration_mag)