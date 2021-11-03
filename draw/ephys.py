import numpy as np
import matplotlib.pyplot as plt



class Raster:
    def __init__(self, spikes:np.ndarray, starts:list, ends:list, ax:plt.Axes=None, color:str='k', lw:float=.5):
        '''
            Raster plot with vertical lines denoting spikes at multiple trials.
        '''
        ax = ax or plt.gca()

        if len(starts) != len(ends):
            raise ValueError('unequal number of starts and ends')

        N = len(starts)
        height = 1/N

        for n, (start, end) in enumerate(zip(starts, ends)):
            trial_spikes = spikes[(spikes > start) & (spikes < end)] - start

            y_0 = 1 - n*height
            y_1 = 1 - (n+1)*height
            ax.vlines(trial_spikes, y_0, y_1, color=color, lw=lw, alpha=1)
            ax.vlines([0, end-start],  y_0, y_1, color='r', lw=1)

        ax.set(yticks=[0, 1], yticklabels=[1, N+1])