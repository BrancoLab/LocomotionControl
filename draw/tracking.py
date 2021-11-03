import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union
from mpl_toolkits.axes_grid1 import make_axes_locatable

from myterial import grey_dark


class Tracking:
    """
        Renders tracking as a 2D trace (by default) or, 
        alternatively, as a scatter or heatmap visualization
    """

    def __init__(
        self,
        x: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        ax: plt.Axes = None,
        **kwargs,
    ):
        ax = ax or plt.gca()
        if isinstance(x, pd.Series):
            for i in np.arange(len(x)):
                ax.plot(
                    x[i],
                    y[i],
                    color=kwargs.pop("color", grey_dark),
                    solid_joinstyle="round",
                    **kwargs,
                )
        else:
            ax.plot(
                x,
                y,
                color=kwargs.pop("color", grey_dark),
                solid_joinstyle="round",
                **kwargs,
            )

    @classmethod
    def scatter(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        s: np.ndarray = None,
        c: np.ndarray = None,
        ax: plt.Axes = None,
        **kwargs,
    ):
        ax = ax or plt.gca()

        if isinstance(x, pd.Series):
            for i in np.arange(len(x)):
                ax.scatter(x[i], y[i], s=s, c=c, **kwargs)
        else:
            ax.scatter(x, y, s=s, c=c, **kwargs)

    @classmethod
    def heatmap(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray = None,  # the variable to heatmap, if None XY occupancy
        ax: plt.Axes = None,
        colorbar: bool = False,
        gridsize:int = 30,
        **kwargs,
    ):
        ax = ax or plt.gca()

        if isinstance(x, pd.Series):
            raise ValueError("Heatmapt plotting cannot accept a pandas Series")
        else:
            H = ax.hexbin(x, y, c, gridsize=gridsize, **kwargs)

            if colorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("bottom", size="5%", pad=0.05)

                ax.figure.colorbar(H, cax=cax, orientation="horizontal")
