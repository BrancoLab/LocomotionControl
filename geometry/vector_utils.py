import numpy as np
from typing import List

from geometry.vector import Vector


def vectors_mean(*vectors: List[Vector]):
    return Vector(*np.mean([[v.x, v.y] for v in vectors], 0))


def smooth_path_vectors(path, window: int = 5) -> List[Vector]:
    """
        Smooths vectors of a path by binning them and
        taking the average vector
    """
    velocity_means, accel_means, tangent_means = [], [], []
    for t in np.arange(0, len(path)):
        t_0 = t - window if t > window else 0
        t_1 = t + window if len(path) - t > window else len(path)

        velocity_means.append(
            vectors_mean(*[path.velocity[i] for i in np.arange(t_0, t_1)])
        )
        accel_means.append(
            vectors_mean(*[path.acceleration[i] for i in np.arange(t_0, t_1)])
        )
        tangent_means.append(
            vectors_mean(*[path.tangent[i] for i in np.arange(t_0, t_1)])
        )

    velocity = Vector.from_list(velocity_means[window:])
    acceleration = Vector.from_list(accel_means[window:])
    tangent = Vector.from_list(tangent_means[window:])

    return velocity, acceleration, tangent
