import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union

from myterial import grey_dark


class Tracking:
    """
        Renders tracking as a 2D trace
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
                    x[i], y[i], color=kwargs.pop("color", grey_dark), **kwargs,
                )
        else:
            ax.plot(
                x, y, color=kwargs.pop("color", grey_dark), **kwargs,
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
