# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from pathlib import Path
import numpy as np

from tpd import recorder

import draw
from data.dbase import db_tables
from fcutils.progress import track
from data import data_utils
from analysis import visuals


folder = Path(r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis")
recorder.start(
    base_folder=folder, folder_name="all_units", timestamp=False
)
# %%
'''
    Save a plot with basic stuff for each unit
'''


# get sessions and tracking
sessions = (db_tables.ValidatedSession * db_tables.Session & 'is_recording=1').fetch('name')
logger.info(f'Found {len(sessions)} recordings')

for session in track(sessions):
    logger.info('Fetching tracking')
    tracking = db_tables.Tracking.get_session_tracking(session, movement=False)
    if tracking.empty:
        logger.warning(f'"{session}" - no tracking')
        continue

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

    # loop over units
    for i, unit in units.iterrows():
        save_name = f'{unit.brain_region}_{unit.mouse_id}_{unit.name}_{unit.unit_id}'
        if (folder / 'all_units' / (save_name + '.png')).exists():
            continue



        break
    a = 1
    break

# %%

# prep data
unit_tracking = data_utils.select_by_indices(tracking, unit.spikes)
unit_tracking["spikes"] = unit.spikes
tracking["firing_rate"] = unit.firing_rate

# %%
axes_lookup = {  # give names to axes
    'tracking': 'A',
    'vel':'B',
    'acc':'B',
    'avel':'B',
    'aacc':'B',
}


# create figure
f = plt.figure(figsize=(20, 12), constrained_layout=True)
_axes = f.subplot_mosaic(
    '''
        AABBCC
        AADDEE
    '''
)
axes = {lk:_axes[name] for lk, name in axes_lookup.items()}



# draw tracking
visuals.plot_bin_x_by_y(
    tracking,
    "firing_rate",
    "speed",
    axes["vel"],
    colors='k',
    bins=20,
    min_count=2,
    s=250,
)


visuals.plot_bin_x_by_y(
    tracking,
    "firing_rate",
    "speed",
    axes["vel"],
    colors='k',
    bins=20,
    min_count=2,
    s=250,
)


# %%
# plot firing rate vs spikes vs spikes kde
import numpy as np
x_0 = 50 * 60
x_range = 5 * 60 + x_0
f, ax = plt.subplots(figsize=(16, 8))



ax.vlines(unit.spikes, ymin=0, ymax=1, alpha=.5, lw=2, color='k')
ax.plot(unit.firing_rate * 33.380967, lw=2, color='salmon')

ax.plot([x_0, x_0 + 60 * 100 / 1000], [-.1, -.1], lw=3, color='k')

ax.set(xlim=[x_0, x_range], xticks=np.arange(x_0, x_range, 250), xticklabels=(np.arange(x_0, x_range, 250)/60).round(0), xlabel='time (s)', ylim=[-1, 6])
# %%
n_seconds = len(tracking.x)/60
n_bins = int(n_seconds * 10)


plt.hist(unit.spikes, bins=n_bins)


# %%
from scipy import stats
frate_window = 25


norm = stats.norm(0, frate_window)
X = np.linspace(norm.ppf(0.0001), norm.ppf(0.9999), frate_window)
kernel = norm.pdf(X)
kernel /= np.sum(kernel)  # normalize area under the curve to 1
kernel = kernel * 1/np.max(kernel)

spikes = np.zeros(100)


spikes[22] = 1
spikes[23] = 1
spikes[50] = 1
spikes[75] = 2


fr = np.convolve(spikes, kernel, mode='same')
plt.vlines(np.where(spikes==1), ymin=0, ymax=1, alpha=.5, lw=2, color='k')

plt.plot(kernel)

plt.hist([22, 23, 50, 75, 75])

plt.plot(fr)
# %%
