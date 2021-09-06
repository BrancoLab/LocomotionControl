import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import numpy as np

from fcutils.plot.figure import clean_axes
from myterial import orange, black, blue_light
from myterial.utils import make_palette

from data.dbase.db_tables import Unit, FiringRate


def plot_unit_firing_rate(unit:pd.Series):
    '''
        For a single unit plot the firing rate at each moment for different firing rate windows
    '''
    if unit.empty:
        raise ValueError('An empty pandas series was passed, perhaps the unit ID was invalid.')

    # get the unit data
    data = (Unit * Unit.Spikes * FiringRate & f'name="{unit.name}"' & f'unit_id={unit.unit_id}').fetch()
    frate_windows = data.firing_rate_std
    n_frate_windows = len(frate_windows)
    palette = make_palette(orange, blue_light, n_frate_windows)
    logger.info(f'Found {n_frate_windows} firing rate windows to plot for unit {unit.unit_id}')

    # create figure
    f, ax = plt.subplots(sigsize=(16, 9))
    f.title(f'Unit {unit.unit_id} firing rate')
    f._save_title(f'unit_{unit.unit_id}_firing_rate')

    # plot spikes
    ax.bar(unit.spikes, np.ones_like(unit.spikes), color=black, label='spike times')

    for frate, color in zip(frate_windows, palette):
        frate_data = data.loc[data.firing_rate_std == frate].iloc[0]
        ax.plot(frate_data.firing_rate, lw=2,  color=color, label=f'kernel std: {frate} ms')


    # cleanup
    clean_axes(f)

    time = np.arange(0, unit.spikes.max(), 60*60*5)
    ax.legend()
    ax.set(xticks=time, xticklables=time/60/60, xlabel='time (min)', ylabel='firing rate')