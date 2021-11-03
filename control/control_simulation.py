import sys

sys.path.append("./")
import matplotlib.pyplot as plt
import numpy as np

from myterial import purple, indigo_dark, salmon_dark, blue_dark

import draw
import control


import geometry.vector_analysis as va

from control.lqr_kinematics import KinematicsLQR

# from control.pure_pursuit import PurePqursuit
from control.config import dt

# initialize car
maxTime = 20.0


# get path
wps = control.paths.Waypoints(use="spline")
path = control.paths.BSpline(wps.x, wps.y, fps=10, degree=3)

# draw path and waypoints
f, ax = plt.subplots(figsize=(6, 10))

# get controller
MODEL = KinematicsLQR(path)
# MODEL = PurePursuit(path)

animation_step = 5
frame = 0
for t in np.arange(0, maxTime, dt):
    # ---------------------------------- control --------------------------------- #
    target_speed = MODEL.step(t)

    # ----------------------------------- plot ----------------------------------- #
    if frame % animation_step == 0:
        plt.cla()

        # draw track
        draw.Tracking(path.x, path.y, lw=3)

        # draw car trajectory
        draw.Tracking(
            MODEL.trajectory.x, MODEL.trajectory.y, lw=1.5, color=salmon_dark
        )
        draw.ControlCar(
            MODEL.vehicle.x,
            MODEL.vehicle.y,
            MODEL.vehicle.theta,
            -MODEL.vehicle.steer,
        )

        # draw car velocity and acceleration vectors
        if len(MODEL.trajectory.x) > 5:
            velocity, _, _, acceleration, _, _ = va.compute_vectors(
                MODEL.trajectory.x[:-3], MODEL.trajectory.y[:-3]
            )

            draw.Arrow(
                MODEL.vehicle.x,
                MODEL.vehicle.y,
                velocity.angle[-1],
                L=10 * velocity.magnitude[-1],
                color=indigo_dark,
            )
            draw.Arrow(
                MODEL.vehicle.x,
                MODEL.vehicle.y,
                acceleration.angle[-1],
                L=15 * acceleration.magnitude[-1] * 10,
                color=purple,
            )

        # draw currently considered trajectory:
        draw.Tracking(
            path.x[
                MODEL.target_trajectory.ind_old : MODEL.target_trajectory.horizon
            ],
            path.y[
                MODEL.target_trajectory.ind_old : MODEL.target_trajectory.horizon
            ],
            lw=3,
            color=blue_dark,
        )

        plt.axis("equal")
        plt.title(
            "LQR (Kinematic): v="
            + str(MODEL.vehicle.speed * 3.6)[:4]
            + "km/h - target:"
            + str(target_speed * 3.6)[:4]
            + "km/h"
        )
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        plt.pause(0.001)
    frame += 1

# ax.legend()
# plt.show()
