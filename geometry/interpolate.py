from typing import Union
import numpy as np


def interpolate_values(x0: float, x1: float, p: float):
    """
        Interplates between two values suchh that when p=0
        the interpolated value is x0 and at p=1 it's x1
    """
    return (1 - p) * x0 + p * x1


def interpolate_at_frame(x: Union[list, np.ndarray], frame: int, p: float):
    """
        Interpolates some quantity at a given frame (element index) based
        on the interpolation value x
    """
    return interpolate_values(x[frame], x[frame + 1], p)
