from typing import Union
import numpy as np


def linear(x0: float, x1: float, p: float):
    """
        Interplates linearly between two values suchh that when p=0
        the interpolated value is x0 and at p=1 it's x1
    """
    return (1 - p) * x0 + p * x1


def step(x0: float, x1: float, p: float):
    """
        Interplates step=wise between two values such that when p<0.5
        the interpolated value is x0 and otherwise it's x1
    """
    return x0 if p < 0.5 else x1


def interpolate_at_frame(
    x: Union[list, np.ndarray], frame: int, p: float, method: str = "linear"
):
    """
        Interpolates some quantity at a given frame (element index) based
        on the interpolation value x
    """
    if method == "linear":
        return linear(x[frame], x[frame + 1], p)
    elif method == "step":
        return step(x[frame], x[frame + 1], p)
    else:
        raise ValueError(f'Unrecognized interpolation method: "{method}"')
