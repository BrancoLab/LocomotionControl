import numpy as np
from typing import Union
import matplotlib.pyplot as plt

from myterial import blue_grey


class Hist:
    def __init__(
        self,
        x: Union[np.ndarray, list],
        bins: int = 25,
        color: str = blue_grey,
        ax: plt.Axes = None,
        alpha: float = 0.5,
        label: str = None,
        **kwargs,
    ):
        ax = ax or plt.gca()

        _, self.bins, _ = ax.hist(
            x,
            bins=bins,
            color=color,
            histtype="step",
            lw=4,
            zorder=5,
            **kwargs,
        )
        ax.hist(
            x,
            bins=bins,
            color=color,
            histtype="stepfilled",
            lw=0,
            zorder=4,
            label=label,
            alpha=alpha,
            **kwargs,
        )


if __name__ == "__main__":

    x = np.random.normal(0, 1, 1000)

    Hist(x, label="test")

    plt.legend()
    plt.show()
