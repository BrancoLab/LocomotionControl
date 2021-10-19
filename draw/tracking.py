import matplotlib.pyplot as plt
import numpy as np

from myterial import grey


class Hairpin:
    """
        Renders an image of the hairpin maze on a matplotlib axis
    """

    _img_path = "draw/hairpin.png"

    def __init__(self, ax: plt.Axes, w_extent: int = 40, h_extent: int = 60):
        """
            Renders an image of the hairpin maze on a matplotlib axis
        """
        image = plt.imread(self._img_path)
        ax.imshow(
            image,
            extent=[0, w_extent, 0, h_extent],
            origin="lower",
            zorder=-100,
        )


class Tracking:
    """
        Renders tracking as a 2D trace
    """

    def __init__(self, ax: plt.Axes, x: np.ndarray, y: np.ndarray, **kwargs):
        ax.plot(
            x,
            y,
            lw=kwargs.pop("lw", 1),
            color=kwargs.pop("color", grey),
            alpha=kwargs.pop("alpha", 1),
        )


if __name__ == "__main__":
    import sys

    sys.path.append("./")

    # from data.dbase import db_tables

    # trk = db_tables.Tracking.get_session_tracking(
    #     sess, body_only=True
    # )

    f, ax = plt.subplots(figsize=(7, 10))

    Hairpin(ax)

    # Tracking(ax, x, y)

    plt.show()
