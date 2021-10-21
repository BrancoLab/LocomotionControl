from typing import Union
import numpy as np

import geometry.vector_analysis as va


class Path:
    """
        Represents an XY trajectory and computes
        relevant stuff on it.
    """

    def __init__(
        self,
        x: Union[list, np.ndarray],
        y: Union[list, np.ndarray],
        theta: Union[list, np.ndarray] = None,
    ):

        self.x = x
        self.y = y

        # compute useful vectors
        (
            self.velocity,
            self.tangent,
            self.normal,
            self.acceleration,
            self.speed,
            self.curvature,
        ) = va.compute_vectors(x, y)

        if theta is None:
            theta = self.tangent.angle
        self.theta = theta

        # compute other useful properties
        self.n_steps = len(x)
        self.distance = np.sum(self.speed)


class GrowingPath:
    """
        Path to which info can be added all the time
    """

    x = []
    y = []
    theta = []

    def update(self, x: float, y: float, theta: float):
        self.x.append(x)
        self.y.append(y)
        self.theta.append(theta)

    def finalize(self) -> Path:
        return Path(self.x, self.y, self.theta)
