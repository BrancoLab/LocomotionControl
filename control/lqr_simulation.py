import sys

sys.path.append("./")
import matplotlib.pyplot as plt
import numpy as np

from myterial import purple, indigo_dark, salmon_dark, blue_dark

import draw
import control


import geometry.vector_analysis as va

from control.lqr_kinematics import KinematicsLQR
from control.config import dt

# initialize car
maxTime = 20.0


# get path
wps = control.paths.Waypoints(use="spline")
path = control.paths.BSpline(wps.x, wps.y, degree=3)

# draw path and waypoints
f, ax = plt.subplots(figsize=(6, 10))


# plt.plot(path.curvature)
# draw.Arrows(path.x[::10], path.y[::10], path.theta[::10], L=2, color='red')
# plt.show()
# raise ValueError

# get controller
LQR = KinematicsLQR(path)


for t in np.arange(0, maxTime, dt):
    # ---------------------------------- control --------------------------------- #
    target_speed = LQR.step()

    # ----------------------------------- plot ----------------------------------- #
    plt.cla()

    # draw track
    draw.Tracking(path.x, path.y, lw=3)

    # draw car trajectory
    draw.Tracking(
        LQR.trajectory.x, LQR.trajectory.y, lw=1.5, color=salmon_dark
    )
    draw.ControlCar(
        LQR.vehicle.x, LQR.vehicle.y, LQR.vehicle.theta, -LQR.vehicle.steer,
    )

    # draw car velocity and acceleration vectors
    if len(LQR.trajectory.x) > 5:
        velocity, _, _, acceleration, _, _ = va.compute_vectors(
            LQR.trajectory.x[:-3], LQR.trajectory.y[:-3]
        )

        draw.Arrow(
            LQR.vehicle.x,
            LQR.vehicle.y,
            velocity.angle[-1],
            L=velocity.magnitude[-1],
            color=indigo_dark,
        )
        draw.Arrow(
            LQR.vehicle.x,
            LQR.vehicle.y,
            acceleration.angle[-1],
            L=acceleration.magnitude[-1] * 10,
            color=purple,
        )

    # draw currently considered trajectory:
    draw.Tracking(
        path.x[LQR.target_trajectory.ind_old : LQR.target_trajectory.horizon],
        path.y[LQR.target_trajectory.ind_old : LQR.target_trajectory.horizon],
        lw=3,
        color=blue_dark,
    )

    plt.axis("equal")
    plt.title(
        "LQR (Kinematic): v="
        + str(LQR.vehicle.speed * 3.6)[:4]
        + "km/h - target:"
        + str(target_speed * 3.6)[:4]
        + "km/h"
    )
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [exit(0) if event.key == "escape" else None],
    )
    plt.pause(0.001)

# ax.legend()
# plt.show()
