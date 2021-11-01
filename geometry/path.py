from typing import Union
import numpy as np

np.seterr(all="ignore")

import geometry.vector_analysis as va
from geometry.vector import Vector


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
        fps: int = 60,
    ):

        self.x = x
        self.y = y
        self.fps = fps

        # compute useful vectors
        (
            self.velocity,
            self.tangent,
            self.normal,
            self.acceleration,
            self.speed,
            self.curvature,
        ) = va.compute_vectors(x, y, fps=fps)

        self.acceleration_mag = self.acceleration.dot(self.tangent)

        if theta is None:
            theta = 180 - self.tangent.angle
        self.theta = theta

        # compute other useful properties
        self.n_steps = len(x)
        self.distance = np.sum(self.speed)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item: Union[str, int]) -> Union[Vector, np.ndarray]:
        if isinstance(item, int):
            raise NotImplementedError(
                "Int indexing should return a slice of the path"
            )

        elif isinstance(item, str):
            return self.__dict__[item]


class GrowingPath:
    """
        Path to which info can be added all the time
    """

    def __init__(self):
        self.x = []
        self.y = []
        self.theta = []
        self.speed = []

    def update(
        self, x: float, y: float, theta: float = None, speed: float = None
    ):
        self.x.append(x)
        self.y.append(y)
        self.theta.append(theta)
        self.speed.append(speed)

    def finalize(self) -> Path:
        return Path(self.x, self.y, self.theta)
