import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import numpy as np

from fcutils.plot.figure import clean_axes
from myterial import orange_dark, black, blue_light
from myterial.utils import make_palette

from data.dbase.db_tables import Unit, FiringRate


def plot_unit_firing_rate(unit:pd.Series, end:int=60):
    '''
        For a single unit plot the firing rate at each moment for different firing rate windows
    '''
    if unit.empty:
        raise ValueError('An empty pandas series was passed, perhaps the unit ID was invalid.')

    # get the unit data
    name = unit['name']
    data = pd.DataFrame((Unit * Unit.Spikes * FiringRate & f'name="{name}"' & f'unit_id={unit.unit_id}').fetch())
    frate_windows = data.firing_rate_std.values
    n_frate_windows = len(frate_windows)
    logger.info(f'Found {n_frate_windows} firing rate windows to plot for unit {unit.unit_id}: {frate_windows}')

    # create figure
    f, axes = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
    f.suptitle(f'Unit {unit.unit_id} firing rate')
    f._save_title = f'unit_{unit.unit_id}_firing_rate'

    # plot spikes
    spikes = unit.spikes[unit.spikes < end * 60]
    axes[0].hist(spikes, bins=end*10, color=[.5, .5, .5], density=True)
    axes[0].scatter(spikes, np.random.uniform(-.05, -.001, size=len(spikes)), color=black, s=25, zorder=1, label='spike times')

    palette = make_palette(orange_dark, blue_light, n_frate_windows)
    for frate, color in zip(frate_windows, palette):
        if frate != 100: continue
        frate_data = data.loc[data.firing_rate_std == frate].iloc[0].firing_rate[:end*60]
        axes[1].plot(frate_data, lw=2,  color=color, label=f'kernel std: {frate} ms')
        # break

    # cleanup
    clean_axes(f)

    time = np.arange(0, (end+1)*60, 60*2)
    axes[1].legend()
    axes[1].set(xticks=time, xticklabels=(time/60).astype(np.int32), xlabel='time (s)', ylabel='firing rate')