# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data.dbase._tracking import calc_angular_velocity

import draw
from data.dbase import db_tables
from loguru import logger
from geometry import Path
from geometry.vector_utils import smooth_path_vectors


# %%

session = 'FC_210721_AAA1110750_hairpin'

# get Tracking
tracking = db_tables.Tracking.get_session_tracking(session, movement=False)

# interpolate vectors
window = 4
path = Path(tracking.x.copy(), tracking.y.copy())
(
    velocity,
    acceleration_vec,
    tangent,
) = smooth_path_vectors(
    path, window=window
) 
path = Path(
    tracking.x[window :], tracking.y[window :]
)


# get units and recording sites
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
path.thetadot =calc_angular_velocity(path.theta)
unit = units.loc[units.brain_region == 'CUN'].iloc[4]
unit_path = path @ unit.spikes


f = plt.figure(figsize=(18, 10))
axes = f.subplot_mosaic(
    '''
        AAABBDD
        AAAbbdd
        AAACCEE
        AAAccee
    '''
)
f.suptitle(f'{unit.brain_region} -- {unit.unit_id}')


draw.Tracking.heatmap(
    unit_path.x,
    unit_path.y,
    bins = 'log',
    mincnt=3,
    gridsize=30,
    ax=axes['A']
)

for ax, var in zip('BCDE', ('speed', 'acceleration_mag', 'theta', 'thetadot')):
    axes[ax].scatter(path[var], unit.firing_rate[window:], alpha=.25, color='k')
    axes[ax].set(title=var)

for ax, var in zip('bcde', ('speed', 'acceleration_mag', 'theta', 'thetadot')):
    draw.Hist(path[var], ax=axes[ax], density=True, alpha=.25, color='k')


# %%
