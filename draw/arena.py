import matplotlib.pyplot as plt
from loguru import logger
import sys
import numpy as np

sys.path.append("./")
from pathlib import Path
import os
from PIL import Image
from skimage.util import invert


from fcutils.plot.figure import clean_axes
from myterial import salmon, blue_light, red_light, green, indigo, teal

from draw.gliphs import Rectangle


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

    def __init__(
        self, ax: plt.Axes = None, set_ax=False, img_path: str = None, **kwargs
    ):
        """
            Renders an image of the hairpin maze on a matplotlib axis
        """
        ax = ax or plt.gca()

        image = self.get_image(img_path)

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

    def get_image(self, img_path: str = None) -> np.ndarray:
        if img_path is not None:
            image = plt.imread(img_path)
        else:
            try:
                image = plt.imread(self.get_image_path())
            except FileNotFoundError:  # use complete path when working in a notebook
                logger.warning("Could not draw ROI image")
                return

        image = image[
            self.px_per_cm * self.y_0 : self.px_per_cm * self.y_1,
            self.px_per_cm * self.x_0 : self.px_per_cm * self.x_1,
        ]
        return image

    @staticmethod
    def get_image_path():
        if Path(Hairpin._img_path_local).exists():
            return Hairpin._img_path_local
        else:
            return Hairpin.image_local_path()

    @staticmethod
    def image_local_path():
        if sys.platform == "darwin":
            return "/Users/federicoclaudi/Documents/Github/LocomotionControl/draw/hairpin.png"
        else:
            return r"C:\Users\Federico\Documents\GitHub\pysical_locomotion\draw\hairpin.png"

    @staticmethod
    def to_txt(scale=1, save_folder: Path = None) -> str:
        """
            Returns a txt representation of the image with symbols
            to denote the walls/empty cells for RL training.
            The scale factor is in cm and it denotes the size of a 'cell'.
        """
        if scale != 1:
            raise NotImplementedError("This case needs checking")

        # load scale and binarize image
        img = Image.open(Hairpin.image_local_path())
        new_width = int(40 * 1 / scale)
        new_height = int(60 * 1 / scale)
        img = img.resize((new_width, new_height))
        img = np.array(img)[:, :, 0]
        img[img > 0] = 1
        arena = invert(img)
        arena[arena == 254] = 0
        arena[arena == 255] = 1
        arena = arena[::-1, :]

        # define location of start and reward
        P_loc = (int(20 * (1 / scale)), int(20 * (1 / scale)))
        G_loc = (int(3 * (1 / scale)), int(56 * (1 / scale)))

        # create text representation
        arena_txt = np.empty(arena.shape, dtype="object")
        arena_txt[arena == 0] = "*"
        arena_txt[arena == 1] = " "
        arena_txt[P_loc[1], P_loc[0]] = "P"
        arena_txt[G_loc[1], G_loc[0]] = "G"

        # write to text

        if save_folder is not None:
            save_path = save_folder / "hairpin.txt"
        else:
            save_path = "hairpin.txt"

        with open(save_path, "w") as fout:
            lines = []
            for rown in range(arena_txt.shape[0]):
                lines.append("".join(arena_txt[rown]) + "\n")
            fout.writelines(lines)

        return arena_txt


class T1(Hairpin):
    """
        First TURN
    """

    # coordinates to crop the image
    x_0 = 8
    x_1 = 24
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
    y_0 = 10
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
    y_0 = 15
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
    y_0 = 10
    y_1 = 47

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
    y_0 = 2
    y_1 = 50

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

    Hairpin.to_txt()

    f, ax = plt.subplots(figsize=(7, 10))

    Hairpin(ax, alpha=0.5)

    # plt.show()
