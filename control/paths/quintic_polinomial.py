from math import sin, cos, atan2
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

import sys

sys.path.append("./")

import draw
from control.paths.waypoints import Waypoints


"""
    Code adapted from: https://github.com/zhm-real/CurvesGenerator
    zhm-real shared code to create different types of paths under an MIT license.
    The logic of the code is left un-affected here, I've just refactored it.
"""


class Trajectory:
    def __init__(self):
        self.t = []
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.a = []
        self.jerk = []


class QuinticPolynomialSegment:
    """
        Fits a Quintic polinomial between an initial and final position
        (position, velocity, acceleration)
    """

    def __init__(self, x0, v0, a0, x1, v1, a1, T):
        A = np.array(
            [
                [T ** 3, T ** 4, T ** 5],
                [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                [6 * T, 12 * T ** 2, 20 * T ** 3],
            ]
        )
        b = np.array(
            [x1 - x0 - v0 * T - a0 * T ** 2 / 2, v1 - v0 - a0 * T, a1 - a0]
        )
        X = np.linalg.solve(A, b)

        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0
        self.a3 = X[0]
        self.a4 = X[1]
        self.a5 = X[2]

    def calc_xt(self, t):
        xt = (
            self.a0
            + self.a1 * t
            + self.a2 * t ** 2
            + self.a3 * t ** 3
            + self.a4 * t ** 4
            + self.a5 * t ** 5
        )

        return xt

    def calc_dxt(self, t):
        dxt = (
            self.a1
            + 2 * self.a2 * t
            + 3 * self.a3 * t ** 2
            + 4 * self.a4 * t ** 3
            + 5 * self.a5 * t ** 4
        )

        return dxt

    def calc_ddxt(self, t):
        ddxt = (
            2 * self.a2
            + 6 * self.a3 * t
            + 12 * self.a4 * t ** 2
            + 20 * self.a5 * t ** 3
        )

        return ddxt

    def calc_dddxt(self, t):
        dddxt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return dddxt


class QuinticPolinomial:
    """
        Fits a Quintic Polinomial to each segment in a series of waypoints
    """

    dt = 0.1  # T tick [s]
    MAX_ACCEL = 50000000.0  # max accel [m/s2]
    MAX_JERK = 10000000.5  # max jerk [m/s3]

    # repeat fits with different durations until we have a valid result
    MIN_T = 1
    MAX_T = 200
    T_STEP = 5

    def __init__(self, waypoints: Waypoints):
        self.t = []
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.a = []
        self.jerk = []

        self.waypoints = waypoints

    def append_to_path(
        self,
        xqp: QuinticPolynomialSegment,
        yqp: QuinticPolynomialSegment,
        T: float,
    ):
        """
            Steps through the results of fitting the polynomial
            to a segment and computes things like velocity and acceleration,
            then it appends everything to the path
        """
        for t in np.arange(0.0, T + self.dt, self.dt):
            self.t.append(t)
            self.x.append(xqp.calc_xt(t))
            self.y.append(yqp.calc_xt(t))

            vx = xqp.calc_dxt(t)
            vy = yqp.calc_dxt(t)
            self.v.append(np.hypot(vx, vy))
            self.yaw.append(atan2(vy, vx))

            ax = xqp.calc_ddxt(t)
            ay = yqp.calc_ddxt(t)
            a = np.hypot(ax, ay)

            if len(self.v) >= 2 and self.v[-1] - self.v[-2] < 0.0:
                a *= -1
            self.a.append(a)

            jx = xqp.calc_dddxt(t)
            jy = yqp.calc_dddxt(t)
            j = np.hypot(jx, jy)

            if len(self.a) >= 2 and self.a[-1] - self.a[-2] < 0.0:
                j *= -1
            self.jerk.append(j)

    def fit(self):
        # iterate over each segment
        for i in range(len(self.waypoints) - 1):
            logger.info(f"Fitting waypoint: {i+1}")
            wp1, wp2 = self.waypoints[i], self.waypoints[i + 1]
            theta1, theta2 = np.radians(wp1.theta), np.radians(wp2.theta)

            # compute vel/acc in each direction
            # s = start, g = goal
            sv_x = wp1.speed * cos(theta1)
            sv_y = wp1.speed * sin(theta1)
            gv_x = wp2.speed * cos(theta2)
            gv_y = wp2.speed * sin(theta2)

            sa_x = wp1.accel * cos(theta1)
            sa_y = wp1.accel * sin(theta1)
            ga_x = wp2.accel * cos(theta2)
            ga_y = wp2.accel * sin(theta2)

            # fit quintic polinomials for X and Y with different durations
            # until we have a good result
            solution_found = False
            T = self.MIN_T
            while not solution_found:
                segment_duration = T  # int(wp2.segment_duration * T)

                # get quantics
                xqp = QuinticPolynomialSegment(
                    wp1.x, sv_x, sa_x, wp2.x, gv_x, ga_x, segment_duration
                )
                yqp = QuinticPolynomialSegment(
                    wp1.y, sv_y, sa_y, wp2.y, gv_y, ga_y, segment_duration
                )

                # check if accel and jerk are within parameters
                path = Trajectory()
                for t in np.arange(0.0, T + self.dt, self.dt):
                    path.t.append(t)
                    path.x.append(xqp.calc_xt(t))
                    path.y.append(yqp.calc_xt(t))

                    vx = xqp.calc_dxt(t)
                    vy = yqp.calc_dxt(t)
                    path.v.append(np.hypot(vx, vy))
                    path.yaw.append(atan2(vy, vx))

                    ax = xqp.calc_ddxt(t)
                    ay = yqp.calc_ddxt(t)
                    a = np.hypot(ax, ay)

                    if len(path.v) >= 2 and path.v[-1] - path.v[-2] < 0.0:
                        a *= -1
                    path.a.append(a)

                    jx = xqp.calc_dddxt(t)
                    jy = yqp.calc_dddxt(t)
                    j = np.hypot(jx, jy)

                    if len(path.a) >= 2 and path.a[-1] - path.a[-2] < 0.0:
                        j *= -1
                    path.jerk.append(j)

                if (
                    max(np.abs(path.a)) <= self.MAX_ACCEL
                    and max(np.abs(path.jerk)) <= self.MAX_JERK
                ):
                    logger.debug(f"Found solution with T={T}")

                    # add results to path
                    self.append_to_path(xqp, yqp, segment_duration)
                    solution_found = True
                else:
                    T += self.T_STEP
                    if T > self.MAX_T:
                        solution_found = True
                        logger.warning("Could not find a valid solution")

            # break
        return self


def simulation(path=QuinticPolinomial):
    """
        Creates an animated plot of a car following the quintic polinomial
    """

    for i in range(len(path.t)):
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        draw.Car(path.x[i], path.y[i], np.degrees(path.yaw[i]), 1.5, 3)
        plt.pause(0.001)
    plt.show()


if __name__ == "__main__":
    import sys

    sys.path.append("./")

    import pandas as pd

    f, ax = plt.subplots(figsize=(7, 10))

    # load and draw tracking data
    from fcutils.path import files

    for n, fp in enumerate(
        files(
            "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/control/",
            "*.h5",
        )
    ):
        tracking = pd.read_hdf(fp, key="hdf")
        tracking.x = 20 - tracking.x + 20
        # draw.Tracking.scatter(tracking.x, tracking.y, c=tracking.speed, vmin=0, vmax=100, cmap='bwr', lw=1, ec='k')
        draw.Tracking(tracking.x, tracking.y, alpha=0.7)

        if n == 5:
            wps = Waypoints.from_tracking(
                tracking.x, tracking.y, tracking.speed, tracking.theta,
            )

    # draw hairpin arena
    draw.Hairpin(ax)

    # draw waypoints
    for wp in wps:
        draw.Arrow(wp.x, wp.y, wp.theta, 2, width=4, color="g")

    # fit and animate quintic polinomial
    # wps = [
    #     Waypoint(20, 40, 270, 1, .2, 5),
    #     Waypoint(20, 30, 270, 3, 0, 5),
    #     Waypoint(15, 5, 180, 3, 0, 5),
    # ]
    # wps = [
    #     Waypoint(x=20.682933975843298, y=36.4433300139396, theta=241.34371213247948, speed=3.4675332536458368, accel=3.4675332536458368, segment_duration=1),
    #     Waypoint(x=20.09675309568283, y=21.35347306956288, theta=274.07352782608245, speed=62.565652699469716, accel=0.7206048508470744, segment_duration=1)
    # ]
    qp = QuinticPolinomial(wps).fit()

    # draw path and iitial/final positoin
    draw.Tracking(qp.x, qp.y, alpha=1, lw=1, color="k")

    draw.Car(
        qp.waypoints[0].x, qp.waypoints[0].y, qp.waypoints[0].theta, 1.5, 3
    )
    draw.Car(
        qp.waypoints[1].x, qp.waypoints[1].y, qp.waypoints[1].theta, 1.5, 3
    )
    simulation(qp)

    plt.show()
