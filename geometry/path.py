from __future__ import annotations

from typing import Union
import numpy as np

np.seterr(all="ignore")

import geometry.vector_analysis as va
from geometry.vector import Vector
from geometry import interpolate


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
        self.distance = np.sum(self.speed) / self.fps

        self.points = np.array([self.x, self.y]).T

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item: Union[str, int]) -> Union[Vector, np.ndarray]:
        if isinstance(item, int):
            return Vector(self.x[item], self.y[item])

        elif isinstance(item, str):
            return self.__dict__[item]

    def interpolate(self, spacing: float = 1) -> Path:
        """
            Interpolates the current path to produce 
            a new path with points 'spacing' apart
        """
        generated = dict(x=[], y=[])
        for n in range(len(self) - 1):
            # get current and next point
            p0 = self[n]
            p1 = self[n + 1]

            # get number of new points
            segment = p1 - p0
            if segment.magnitude <= spacing:
                if n > 0:
                    prev = Vector(generated["x"][-1], generated["y"][-1])
                    if (prev - p0).magnitude >= spacing:
                        generated["x"].append(p0.x)
                        generated["y"].append(p0.y)
                    else:
                        continue
                else:
                    generated["x"].append(p0.x)
                    generated["y"].append(p0.y)
            else:
                n_new = int(np.ceil(segment.magnitude / spacing))

                # create new points
                for p in np.linspace(0, 1, n_new):
                    if n > 0 and p == 0:
                        continue  # avoid doubling
                    generated["x"].append(interpolate.linear(p0.x, p1.x, p))
                    generated["y"].append(interpolate.linear(p0.y, p1.y, p))
        return Path(generated["x"], generated["y"])

    def downsample(self, spacing: float = 1) -> Path:
        """
            Downsamples the path keeping only points that are spacing apart
        """
        downsampled = dict(x=[], y=[])
        for n in range(len(self)):
            if n == 0:
                downsampled["x"].append(self[0].x)
                downsampled["y"].append(self[0].y)
            else:
                # get current and prev point
                p0 = Vector(downsampled["x"][-1], downsampled["y"][-1])
                p1 = self[n]

                # if distance > spacing, keep point
                if (p1 - p0).magnitude > spacing:
                    downsampled["x"].append(p1.x)
                    downsampled["y"].append(p1.y)
        return Path(downsampled["x"], downsampled["y"])


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
