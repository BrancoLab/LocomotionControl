import matplotlib.pyplot as plt
import numpy as np

from myterial import grey_dark




class Tracking:
    """
        Renders tracking as a 2D trace
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, ax: plt.Axes = None, **kwargs
    ):
        ax = ax or plt.gca()
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
        ax.scatter(x, y, s=s, c=c, **kwargs)


