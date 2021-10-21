import matplotlib.pyplot as plt

import sys

sys.path.append("./")
from pathlib import Path
import os
from draw.gliphs import Rectangle

from fcutils.plot.figure import clean_axes
from myterial import salmon, blue_light, red_light, green, indigo, teal


class Hairpin:
    """
        Renders an image of the hairpin maze on a matplotlib axis
    """

    _img_path = Path(os.getcwd()).parent / "draw/hairpin.png"
    _img_path_local = "draw/hairpin.png"

    px_per_cm = 45

    # coordinates to crop the image
    x_0 = 0
    x_1 = 40
    y_0 = 0
    y_1 = 60

    def __init__(self, ax: plt.Axes = None, set_ax=False, **kwargs):
        """
            Renders an image of the hairpin maze on a matplotlib axis
        """
        ax = ax or plt.gca()
        try:
            image = plt.imread(self._img_path_local)
        except FileNotFoundError:
            image = plt.imread(self._img_path)

        # raise ValueError(image.shape[1] / 40, image.shape[0] / 60)
        image = image[
            self.px_per_cm * self.y_0 : self.px_per_cm * self.y_1,
            self.px_per_cm * self.x_0 : self.px_per_cm * self.x_1,
        ]

        ax.imshow(
            image,
            extent=[self.x_0, self.x_1, self.y_0, self.y_1],
            origin="lower",
            zorder=-100,
            **kwargs,
        )

        if set_ax:
            ax.set(
                xlim=[self.x_0, self.x_1],
                ylim=[self.y_0, self.y_1],
                xlabel="cm",
                ylabel="cm",
                xticks=[self.x_0, self.x_1],
                yticks=[self.y_0, self.y_1],
            )
        clean_axes(ax.figure)
        ax.axis("equal")


class T1(Hairpin):
    """
        First TURN
    """

    # coordinates to crop the image
    x_0 = 8
    x_1 = 24
    y_0 = 0
    y_1 = 25

    def __init__(self, ax: plt.Axes = None, shade=False, **kwargs):
        super().__init__(ax, **kwargs)
        if shade:
            Rectangle(
                self.x_0,
                self.x_1,
                self.y_0,
                self.y_1,
                ax=ax,
                color=salmon,
                alpha=0.2,
                lw=0,
            )


class T2(Hairpin):
    """
        Second TURN
    """

    # coordinates to crop the image
    x_0 = 8
    x_1 = 32
    y_0 = 24
    y_1 = 52

    def __init__(self, ax: plt.Axes = None, shade=False, **kwargs):
        super().__init__(ax, **kwargs)
        if shade:
            Rectangle(
                self.x_0,
                self.x_1,
                self.y_0,
                self.y_1,
                ax=ax,
                color=blue_light,
                alpha=0.2,
                lw=0,
            )


class T3(Hairpin):
    """
        Third TURN
    """

    # coordinates to crop the image
    x_0 = 24
    x_1 = 40
    y_0 = 0
    y_1 = 35

    def __init__(self, ax: plt.Axes = None, shade=False, **kwargs):
        super().__init__(ax, **kwargs)
        if shade:
            Rectangle(
                self.x_0,
                self.x_1,
                self.y_0,
                self.y_1,
                ax=ax,
                color=red_light,
                alpha=0.2,
                lw=0,
            )


class T4(Hairpin):
    """
        Third TURN
    """

    # coordinates to crop the image
    x_0 = 0
    x_1 = 40
    y_0 = 33
    y_1 = 60

    def __init__(self, ax: plt.Axes = None, shade=False, **kwargs):
        super().__init__(ax, **kwargs)
        if shade:
            Rectangle(
                self.x_0,
                self.x_1,
                self.y_0,
                self.y_1,
                ax=ax,
                color=indigo,
                alpha=0.2,
                lw=0,
            )


class S1(Hairpin):
    """
        straight segment
    """

    # coordinates to crop the image
    x_0 = 32
    x_1 = 40
    y_0 = 25
    y_1 = 45

    def __init__(self, ax: plt.Axes = None, shade=False, **kwargs):
        super().__init__(ax, **kwargs)
        if shade:
            Rectangle(
                self.x_0,
                self.x_1,
                self.y_0,
                self.y_1,
                ax=ax,
                color=green,
                alpha=0.15,
                lw=0,
            )


class S2(Hairpin):
    """
        straight segment
    """

    # coordinates to crop the image
    x_0 = 0
    x_1 = 8
    y_0 = 5
    y_1 = 45

    def __init__(self, ax: plt.Axes = None, shade=False, **kwargs):
        super().__init__(ax, **kwargs)
        if shade:
            Rectangle(
                self.x_0,
                self.x_1,
                self.y_0,
                self.y_1,
                ax=ax,
                color=teal,
                alpha=0.15,
                lw=0,
            )


def ROI(roi_name: str, *args, **kwargs):
    if roi_name == "T1":
        return T1(*args, **kwargs)
    elif roi_name == "T2":
        return T2(*args, **kwargs)
    elif roi_name == "T3":
        return T3(*args, **kwargs)
    elif roi_name == "T4":
        return T4(*args, **kwargs)
    elif roi_name == "S1":
        return S1(*args, **kwargs)
    elif roi_name == "S2":
        return S2(*args, **kwargs)
    else:
        raise ValueError(roi_name)


if __name__ == "__main__":
    import sys

    sys.path.append("./")

    # from data.dbase import db_tables

    # trk = db_tables.Tracking.get_session_tracking(
    #     sess, body_only=True
    # )

    f, ax = plt.subplots(figsize=(7, 10))

    Hairpin(ax, alpha=0.5)

    T1()
    T2()
    T3()
    T4()
    S1()
    S2()

    ax.set(xlim=[0, 40], ylim=[0, 60])

    plt.show()
