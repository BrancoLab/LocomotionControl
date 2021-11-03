from __future__ import annotations

import sys

sys.path.append("./")

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
                n_new = int(np.floor(segment.magnitude / spacing))

                # create new points
                for p in np.linspace(0, 1, n_new):
                    if n > 0 and p == 0:
                        continue  # avoid doubling
                    generated["x"].append(interpolate.linear(p0.x, p1.x, p))
                    generated["y"].append(interpolate.linear(p0.y, p1.y, p))
        return Path(generated["x"], generated["y"])

    def downsample(self, spacing: float = 1) -> Path:
        """
            Downsamples the path keeping only points that are spacing apart.
            It downsamples the path by selecting points that are spaced
            along the path: the spacing reflects the path-length between two
            points along the original path, even though they might be very close in 
            euclidean terms.
        """
        downsampled = dict(x=[], y=[])
        for n in range(len(self)):
            if n == 0:
                downsampled["x"].append(self[0].x)
                downsampled["y"].append(self[0].y)
                last_distance = 0
            else:
                # get path distance until current point
                curr_distance = np.sum(self.speed[:n]) / self.fps

                if curr_distance - last_distance > spacing:
                    downsampled["x"].append(self[n].x)
                    downsampled["y"].append(self[n].y)
                    last_distance = curr_distance
        return Path(downsampled["x"], downsampled["y"])

    def downsample_euclidean(self, spacing: float = 1) -> Path:
        """
            Downsamples the path keeping only points that are spacing apart.
            This function looks at the euclidean distance between points, 
            ignores the path length distance between them
        """
        downsampled = dict(x=[], y=[])
        for n in range(len(self)):
            if n == 0:
                downsampled["x"].append(self[0].x)
                downsampled["y"].append(self[0].y)
            else:
                p0 = Vector(downsampled["x"][-1], downsampled["y"][-1])
                p1 = self[n]

                # if distance > spacing
                if (p1 - p0).magnitude >= spacing:
                    # the distance between the two could be > spacing
                    # get a new point at the right distance
                    vec = p1 - p0
                    downsampled["x"].append(
                        p0.x + spacing * np.cos(np.radians(vec.angle))
                    )
                    downsampled["y"].append(
                        p0.y + spacing * np.sin(np.radians(vec.angle))
                    )
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import draw

    # x = np.linspace(0, 30, 100)
    # y = np.sin(x)

    c = np.linspace(0, 1.5 * np.pi, 100)
    x = c
    y = np.sin(c)

    path = Path(x, y)
    downsampled = path.downsample(1)
    downsampled_eu = path.downsample_euclidean(1)
    upsampled = downsampled.interpolate(0.25)

    draw.Tracking(path.x, path.y, color="k", alpha=0.5)
    draw.Tracking.scatter(
        path.x, path.y, color="k", label="original", alpha=0.5
    )

    draw.Tracking(downsampled.x, downsampled.y, color="r")
    draw.Tracking.scatter(
        downsampled.x, downsampled.y, color="r", label="downsampled"
    )

    draw.Tracking(upsampled.x, upsampled.y, color="b")
    draw.Tracking.scatter(
        upsampled.x, upsampled.y, color="b", label="resampled"
    )

    draw.Tracking(downsampled_eu.x, downsampled_eu.y, color="g")
    draw.Tracking.scatter(
        downsampled_eu.x, downsampled_eu.y, color="g", label="downsampled eucl"
    )

    plt.legend()
    plt.show()
