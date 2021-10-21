import numpy as np
from typing import Tuple

import sys

sys.path.append("./")

# ! IMPLEMENT CONSISTENCY FOR NORM SUCH THAT IT ALWAYS STAYS ON THE SAME SIDE

np.seterr(all="ignore")

from geometry.vector import Vector


def compute_vectors(x: np.ndarray, y: np.ndarray) -> Tuple[Vector, np.ndarray]:
    """
        Given the X and Y position at each frame -

        Compute vectors:
            i. velocity vector
            ii. unit tangent 
            iii. unit norm
            iv. acceleration

        and scalar quantities:
            i. speed
            ii. curvature
        
        See: https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
    """
    # compute velocity vector
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    velocity = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])

    # compute scalr speed vector
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

    # get unit tangent vector
    tangent = np.array([1 / ds_dt] * 2).transpose() * velocity

    # get unit normal vector
    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]

    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    dT_dt = np.array(
        [
            [deriv_tangent_x[i], deriv_tangent_y[i]]
            for i in range(deriv_tangent_x.size)
        ]
    )

    length_dT_dt = np.sqrt(
        deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y
    )

    normal = np.array([1 / length_dT_dt] * 2).transpose() * dT_dt

    # get acceleration and curvature
    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = (
        np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
        / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    )
    t_component = np.array([d2s_dt2] * 2).transpose()
    n_component = np.array([curvature * ds_dt * ds_dt] * 2).transpose()

    acceleration = t_component * tangent + n_component * normal

    return (
        Vector(velocity),
        Vector(tangent),
        Vector(normal),
        Vector(acceleration),
        ds_dt,
        curvature,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import sys

    sys.path.append("./")

    import draw

    x = np.linspace(0, 5, 100)
    y = np.sin(2 * x)
    y[(x > 2) & (x < 2.75)] = -0.75

    (
        velocity,
        tangent,
        normal,
        acceleration,
        speed,
        curvature,
    ) = compute_vectors(x, y)

    f, axes = plt.subplots(nrows=2, sharex=False, figsize=(12, 8))

    axes[0].scatter(x, y)

    draw.Arrows(
        x[::7],
        y[::7],
        tangent.angle[::7],
        ax=axes[0],
        L=0.25,
        color="r",
        label="tangent",
    )
    draw.Arrows(
        x[::7],
        y[::7],
        normal.angle[::7],
        ax=axes[0],
        L=0.25,
        color="g",
        label="normal",
    )
    draw.Arrows(
        x[::7],
        y[::7],
        acceleration.angle[::7],
        ax=axes[0],
        L=0.25,
        color="m",
        label="acceleration",
    )

    axes[1].plot(x, speed, label="speed")
    axes[1].plot(x, curvature, label="curvature")

    axes[0].legend()
    axes[1].legend()
    plt.show()
