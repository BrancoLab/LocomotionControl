"""
Code adapted from: https://github.com/zhm-real/CurvesGenerator
(under MIT license)
to fit a bezier path to a list of points.

bezier path
author: Atsushi Sakai(@Atsushi_twi)
modified: huiming zhou
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def calc_bezier_path(control_points, n_points=100):
    traj = []

    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))

    return np.array(traj)


def Comb(n, i, t):
    return comb(n, i) * t ** i * (1 - t) ** (n - i)


def bezier(t, control_points):
    n = len(control_points) - 1
    return np.sum(
        [Comb(n, i, t) * control_points[i] for i in range(n + 1)], axis=0
    )


if __name__ == "__main__":
    pts = np.array([[0, 0], [1, 0], [2, 2], [3, 6]])
    path = calc_bezier_path(pts, 100)

    plt.scatter(pts[:, 0], pts[:, 1])
    plt.plot(path[:, 0], path[:, 1])
    plt.show()
