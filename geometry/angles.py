import numpy as np

from fcutils.maths import derivative

import geometry


def angular_derivative(angles: np.ndarray) -> np.ndarray:
    """
        Takes the deriative of an angular variable (in degrees)
    """
    # convert to radians and take derivative
    rad = np.unwrap(np.deg2rad(angles))
    diff = derivative(rad)
    return np.rad2deg(diff)


def orientation(
    x_0: np.ndarray,
    y_0: np.ndarray,
    x_1: np.ndarray,
    y_1: np.ndarray,
    smooth: bool = True,
) -> np.ndarray:
    """
        Given two sets of XY coordinates for e.g. two bodyparts, 
        compute the angle going from one to the other, at all frames
        (they must be of the same length)
    """
    delta_path = geometry.Path(x_1 - x_0, y_1 - y_0)
    if smooth:
        delta_path = delta_path.smooth()

    return delta_path.theta
