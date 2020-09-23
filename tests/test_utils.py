import numpy as np

from proj.utils.misc import cartesian_to_polar, polar_to_cartesian


def test_coord_transf():
    X = [1, 1, 0, 1, -1, 0]
    Y = [0, 1, 1, -1, 0, -1]
    R = [1, np.sqrt(2), 1, np.sqrt(2), 1, 1]
    angles = np.radians([0, 45, 90, -45, 180, -90])

    for x, y, rr, ang in zip(X, Y, R, angles):
        # cartesian to polar
        r, gamma = cartesian_to_polar(x, y)

        if r != rr:
            raise ValueError(f"Cart->Pol expected r: {rr} got: {r}")

        if gamma != ang:
            raise ValueError(f"Cart->Pol expected gamma: {ang} got: {gamma}")

        # polar to cartesian
        x2, y2 = polar_to_cartesian(r, gamma)
        x2 = np.round(x2, 2)
        y2 = np.round(y2, 2)

        if x2 != x or y2 != y:
            raise ValueError(f"Pol->Cart expected {(x, y)} but got {(x2, y2)}")
