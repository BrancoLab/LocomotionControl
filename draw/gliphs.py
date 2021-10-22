import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle as Rectangle_patch
from matplotlib.patches import Polygon as Polygon_patch

from typing import Union

from myterial import blue_grey_dark, grey_dark

from control.config import (
    wheelbase,
    wheeldist,
    v_w,
    r_b,
    r_f,
    t_r,
    t_w,
)


class Arrow:
    """
        Draws an arrow at a point and angle
    """

    def __init__(
        self,
        x: float,
        y: float,
        theta: float,  # in degrees
        L: float = 1,  # length
        width: float = 4,
        color: str = blue_grey_dark,
        zorder: int = 100,
        ax: plt.Axes = None,
        outline: bool = False,  # draw a larger darker arrow under the main one
        label: str = None,
        alpha: float = 1,
        **kwargs,
    ):
        if outline:
            widths = [width + 1, width]
            colors = ["k", color]
            labels = [None, label]
        else:
            widths = [width]
            colors = [color]
            labels = [label]

        ax = ax or plt.gca()

        # compute arrow position
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

        # draw
        for width, color, label in zip(widths, colors, labels):
            ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                color=color,
                linewidth=width,
                zorder=zorder,
                label=label,
                alpha=alpha,
                **kwargs,
            )
            ax.plot(
                [x_hat_start, x_hat_end_L],
                [y_hat_start, y_hat_end_L],
                color=color,
                linewidth=width,
                zorder=zorder,
                alpha=alpha,
            )
            ax.plot(
                [x_hat_start, x_hat_end_R],
                [y_hat_start, y_hat_end_R],
                color=color,
                linewidth=width,
                zorder=zorder,
                alpha=alpha,
            )


class Arrows:
    """
        Draws an arrow at a point and angle
    """

    def __init__(
        self,
        x: list,
        y: list,
        theta: list,  # in degrees
        label=None,
        step: int = 1,  # draw arrow every step
        L: Union[list, np.ndarray, float] = 1,
        color: Union[str, list] = "k",
        **kwargs,
    ):

        # make sure L and color are iterable
        if isinstance(L, (int, float)):
            L = [L] * len(x)
        if isinstance(color, str):
            color = [color] * len(x)

        # draw each arrow
        for i in range(len(x)):
            if i > 0:
                label = None
            if i % step == 0:
                Arrow(
                    x[i],
                    y[i],
                    theta[i],
                    label=label,
                    L=L[i],
                    color=color[i],
                    **kwargs,
                )


class Dot:
    def __init__(
        self,
        x: float,
        y: float,
        ax: plt.Axes = None,
        zorder=100,
        s=100,
        color="k",
        **kwargs,
    ):
        ax = ax or plt.gca()
        ax.scatter(x, y, zorder=zorder, s=s, color=color, **kwargs)


class Car:
    def __init__(
        self,
        x: float,
        y: float,
        theta: float,  # in degrees
        w: int = 1.5,
        L: int = 3,
        ax: plt.Axes = None,
    ):
        ax = ax or plt.gca()

        theta = np.radians(theta)
        theta_B = np.pi + theta

        xB = x + L / 4 * np.cos(theta_B)
        yB = y + L / 4 * np.sin(theta_B)

        theta_BL = theta_B + np.pi / 2
        theta_BR = theta_B - np.pi / 2

        x_BL = xB + w / 2 * np.cos(theta_BL)  # Bottom-Left vertex
        y_BL = yB + w / 2 * np.sin(theta_BL)
        x_BR = xB + w / 2 * np.cos(theta_BR)  # Bottom-Right vertex
        y_BR = yB + w / 2 * np.sin(theta_BR)

        x_FL = x_BL + L * np.cos(theta)  # Front-Left vertex
        y_FL = y_BL + L * np.sin(theta)
        x_FR = x_BR + L * np.cos(theta)  # Front-Right vertex
        y_FR = y_BR + L * np.sin(theta)

        ax.plot(
            [x_BL, x_BR, x_FR, x_FL, x_BL],
            [y_BL, y_BR, y_FR, y_FL, y_BL],
            linewidth=1,
            color="black",
        )

        Arrow(x, y, np.degrees(theta), L / 2, color="black")


class ControlCar:
    def __init__(self, x, y, theta, steer, color="black", ax: plt.Axes = None):
        ax = ax or plt.gca()

        car = np.array(
            [
                [-r_b, -r_b, r_f, r_f, -r_b],
                [v_w / 2, -v_w / 2, -v_w / 2, v_w / 2, v_w / 2],
            ]
        )

        wheel = np.array(
            [
                [-t_r, -t_r, t_r, t_r, -t_r],
                [t_w / 2, -t_w / 2, -t_w / 2, t_w / 2, t_w / 2],
            ]
        )

        rlWheel = wheel.copy()
        rrWheel = wheel.copy()
        frWheel = wheel.copy()
        flWheel = wheel.copy()

        Rot1 = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        Rot2 = np.array(
            [[np.cos(steer), np.sin(steer)], [-np.sin(steer), np.cos(steer)]]
        )

        frWheel = np.dot(Rot2, frWheel)
        flWheel = np.dot(Rot2, flWheel)

        frWheel += np.array([[wheelbase], [-wheeldist / 2]])
        flWheel += np.array([[wheelbase], [wheeldist / 2]])
        rrWheel[1, :] -= wheeldist / 2
        rlWheel[1, :] += wheeldist / 2

        frWheel = np.dot(Rot1, frWheel)
        flWheel = np.dot(Rot1, flWheel)

        rrWheel = np.dot(Rot1, rrWheel)
        rlWheel = np.dot(Rot1, rlWheel)
        car = np.dot(Rot1, car)

        frWheel += np.array([[x], [y]])
        flWheel += np.array([[x], [y]])
        rrWheel += np.array([[x], [y]])
        rlWheel += np.array([[x], [y]])
        car += np.array([[x], [y]])

        ax.plot(car[0, :], car[1, :], color)
        ax.plot(frWheel[0, :], frWheel[1, :], color, lw=4)
        ax.plot(rrWheel[0, :], rrWheel[1, :], color, lw=4)
        ax.plot(flWheel[0, :], flWheel[1, :], color, lw=4)
        ax.plot(rlWheel[0, :], rlWheel[1, :], color, lw=4)

        Arrow(x, y, np.degrees(theta), L=0.5 * wheelbase, color=color, ax=ax)


class Rectangle:
    def __init__(
        self,
        x_0,
        x_1,
        y_0,
        y_1,
        ax: plt.Axes = None,
        color=blue_grey_dark,
        **kwargs,
    ):
        ax = ax or plt.gca()
        rect = Rectangle_patch(
            (x_0, y_0), x_1 - x_0, y_1 - y_0, color=color, **kwargs
        )
        ax.add_patch(rect)


class Polygon:
    def __init__(
        self, *points, ax: plt.Axes = None, color=grey_dark, **kwargs,
    ):
        """
            Given a list of tuples/lists of XY coordinates of each point, 
            this class draws a polygon
        """
        ax = ax or plt.gca()

        xy = np.vstack(points)

        patch = Polygon_patch(xy, color=color, **kwargs)
        ax.add_patch(patch)


if __name__ == "__main__":
    from numpy.random import uniform

    X = uniform(0, 10, 5)
    Y = uniform(0, 20, 5)
    T = uniform(0, 360, 5)

    f, ax = plt.subplots(figsize=(7, 10))

    for x, y, t in zip(X, Y, T):
        Arrow(ax, x, y, t, 2)

    plt.show()
