import matplotlib.pyplot as plt
import numpy as np

from myterial import blue_grey_dark


class Arrow:
    """
        Draws an arrow at a point and angle
    """

    def __init__(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        theta: float,  # in degrees
        L: float = 0.1,  # length
        width: float = 4,
        color: str = blue_grey_dark,
        zorder: int = 100,
    ):
        theta = np.radians(theta)
        angle = np.deg2rad(30)
        d = 0.5 * L

        x_start = x
        y_start = y
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)

        theta_hat_L = theta + np.pi - angle
        theta_hat_R = theta + np.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            color=color,
            linewidth=width,
            zorder=zorder,
        )
        ax.plot(
            [x_hat_start, x_hat_end_L],
            [y_hat_start, y_hat_end_L],
            color=color,
            linewidth=width,
            zorder=zorder,
        )
        ax.plot(
            [x_hat_start, x_hat_end_R],
            [y_hat_start, y_hat_end_R],
            color=color,
            linewidth=width,
            zorder=zorder,
        )


if __name__ == "__main__":
    from numpy.random import uniform

    X = uniform(0, 10, 5)
    Y = uniform(0, 20, 5)
    T = uniform(0, 360, 5)

    f, ax = plt.subplots(figsize=(7, 10))

    for x, y, t in zip(X, Y, T):
        Arrow(ax, x, y, t, 2)

    plt.show()
