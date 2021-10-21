"""
    Code adapted from: https://github.com/zhm-real/MotionPlanning
    zhm-real shared code to create different types of paths under an MIT license.
    The logic of the code is left un-affected here, I've just refactored it.
"""

import sys
import math
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

sys.path.append("./")

from control.config import (
    wheelbase,
    dt,
    max_iteration,
    eps,
    matrix_q,
    matrix_r,
    state_size,
    max_acceleration,
    max_steer_angle,
    max_speed,
)
from control.paths.utils import pi_2_pi


class Vehicle:
    """
        Represents the car and its current state
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        speed: float = 0.0,
    ):
        self.x = x
        self.y = y
        self.theta = np.radians(theta)
        self.speed = speed
        self.lateral_error = 0.0
        self.theta_error = 0.0
        self.steer = 0.0

    def update_state(
        self,
        steering_angle: float,
        acceleration: float,
        lateral_error: float,
        theta_error: float,
    ):
        """
        update states of vehicle
        :param theta_error: theta error to ref trajectory
        :param lateral_error: lateral error to ref trajectory
        :param steering_angle: steering angle [rad]
        :param a: acceleration [m / s^2]
        """

        steering_angle, acceleration = self.normalize_input(
            steering_angle, acceleration
        )

        self.steer = steering_angle
        self.x += self.speed * math.cos(self.theta) * dt
        self.y += self.speed * math.sin(self.theta) * dt
        self.theta += self.speed / wheelbase * math.tan(steering_angle) * dt
        self.lateral_error = lateral_error
        self.theta_error = theta_error

        self.speed += acceleration * dt
        self.speed = self.normalize_output(self.speed)

    @staticmethod
    def normalize_input(
        steering: float, acceleration: float
    ) -> Tuple[float, float]:
        """
        regulate steering to : - max_steer_angle ~ max_steer_angle
        regulate acceleration to : - max_acceleration ~ max_acceleration
        :param steering: steering angle [rad]
        :param acceleration: acceleration [m / s^2]
        :return: regulated steering and acceleration
        """

        if steering < -1.0 * max_steer_angle:
            steering = -1.0 * max_steer_angle

        if steering > 1.0 * max_steer_angle:
            steering = 1.0 * max_steer_angle

        if acceleration < -1.0 * max_acceleration:
            acceleration = -1.0 * max_acceleration

        if acceleration > 1.0 * max_acceleration:
            acceleration = 1.0 * max_acceleration

        return steering, acceleration

    @staticmethod
    def normalize_output(speed: float):
        """
        regulate v to : -max_speed ~ max_speed
        :param v: calculated speed [m / s]
        :return: regulated speed
        """

        if speed < -1.0 * max_speed:
            speed = -1.0 * max_speed

        if speed > 1.0 * max_speed:
            speed = 1.0 * max_speed

        return speed


class TrajectoryAnalyzer:
    def __init__(self, x, y, theta, k):
        self.x_ = x
        self.y_ = y
        self.theta_ = np.radians(theta)
        self.k_ = k

        self.ind_old = 0
        self.ind_end = len(x)

    def ToTrajectoryFrame(self, vehicle_state):
        """
        errors to trajectory frame
        theta_e = theta_vehicle - theta_ref_path
        e_cg = lateral distance of center of gravity (cg) in frenet frame
        :param vehicle_state: vehicle state (class Vehicle)
        :return: theta_e, e_cg, theta_ref, k_ref
        """

        x_cg = vehicle_state.x
        y_cg = vehicle_state.y
        theta = vehicle_state.theta

        # calc nearest point in ref path
        dx = [x_cg - ix for ix in self.x_[self.ind_old : self.ind_end]]
        dy = [y_cg - iy for iy in self.y_[self.ind_old : self.ind_end]]

        ind_add = int(np.argmin(np.hypot(dx, dy)))
        dist = math.hypot(dx[ind_add], dy[ind_add])

        # calc lateral relative position of vehicle to ref path
        vec_axle_rot_90 = np.array(
            [
                [math.cos(theta + math.pi / 2.0)],
                [math.sin(theta + math.pi / 2.0)],
            ]
        )

        vec_path_2_cg = np.array([[dx[ind_add]], [dy[ind_add]]])

        if np.dot(vec_axle_rot_90.T, vec_path_2_cg) > 0.0:
            e_cg = 1.0 * dist  # vehicle on the right of ref path
        else:
            e_cg = -1.0 * dist  # vehicle on the left of ref path

        # calc theta error: theta_e = theta_vehicle - theta_ref
        self.ind_old += ind_add
        theta_ref = self.theta_[self.ind_old]
        theta_e = pi_2_pi(theta - theta_ref)

        # calc ref curvature
        k_ref = self.k_[self.ind_old]

        return theta_e, e_cg, theta_ref, k_ref


class LatController:
    """
    Lateral Controller using LQR
    """

    def ComputeControlCommand(self, vehicle_state, ref_trajectory):
        """
        calc lateral control command.
        :param vehicle_state: vehicle state
        :param ref_trajectory: reference trajectory (analyzer)
        :return: steering angle (optimal u), theta_e, e_cg
        """

        e_cg_old = vehicle_state.lateral_error
        theta_e_old = vehicle_state.theta_error

        theta_e, e_cg, theta_ref, k_ref = ref_trajectory.ToTrajectoryFrame(
            vehicle_state
        )

        matrix_ad_, matrix_bd_ = self.UpdateMatrix(vehicle_state)

        matrix_state_ = np.zeros((state_size, 1))
        matrix_r_ = np.diag(matrix_r)
        matrix_q_ = np.diag(matrix_q)

        matrix_k_ = self.SolveLQRProblem(
            matrix_ad_, matrix_bd_, matrix_q_, matrix_r_, eps, max_iteration
        )

        matrix_state_[0][0] = e_cg
        matrix_state_[1][0] = (e_cg - e_cg_old) / dt
        matrix_state_[2][0] = theta_e
        matrix_state_[3][0] = (theta_e - theta_e_old) / dt

        steer_angle_feedback = -(matrix_k_ @ matrix_state_)[0][0]

        steer_angle_feedforward = self.ComputeFeedForward(k_ref)

        steer_angle = steer_angle_feedback + steer_angle_feedforward

        return steer_angle, theta_e, e_cg

    @staticmethod
    def ComputeFeedForward(ref_curvature):
        """
        calc feedforward control term to decrease the steady error.
        :param ref_curvature: curvature of the target point in ref trajectory
        :return: feedforward term
        """

        steer_angle_feedforward = wheelbase * ref_curvature

        return steer_angle_feedforward

    @staticmethod
    def SolveLQRProblem(A, B, Q, R, tolerance, max_num_iteration):
        """
        iteratively calculating feedback matrix K
        :param A: matrix_a_
        :param B: matrix_b_
        :param Q: matrix_q_
        :param R: matrix_r_
        :param tolerance: lqr_eps
        :param max_num_iteration: max_iteration
        :return: feedback matrix K
        """

        assert (
            np.size(A, 0) == np.size(A, 1)
            and np.size(B, 0) == np.size(A, 0)
            and np.size(Q, 0) == np.size(Q, 1)
            and np.size(Q, 0) == np.size(A, 1)
            and np.size(R, 0) == np.size(R, 1)
            and np.size(R, 0) == np.size(B, 1)
        ), "LQR solver: one or more matrices have incompatible dimensions."

        M = np.zeros((np.size(Q, 0), np.size(R, 1)))

        AT = A.T
        BT = B.T
        MT = M.T

        P = Q
        num_iteration = 0
        diff = math.inf

        while num_iteration < max_num_iteration and diff > tolerance:
            num_iteration += 1
            P_next = (
                AT @ P @ A
                - (AT @ P @ B + M)
                @ np.linalg.pinv(R + BT @ P @ B)
                @ (BT @ P @ A + MT)
                + Q
            )

            # check the difference between P and P_next
            diff = (abs(P_next - P)).max()
            P = P_next

        if num_iteration >= max_num_iteration:
            print(
                "LQR solver cannot converge to a solution",
                "last consecutive result diff is: ",
                diff,
            )

        K = np.linalg.inv(BT @ P @ B + R) @ (BT @ P @ A + MT)

        return K

    @staticmethod
    def UpdateMatrix(vehicle_state):
        """
        calc A and b matrices of linearized, discrete system.
        :return: A, b
        """

        v = vehicle_state.speed

        matrix_ad_ = np.zeros(
            (state_size, state_size)
        )  # time discrete A matrix

        matrix_ad_[0][0] = 1.0
        matrix_ad_[0][1] = dt
        matrix_ad_[1][2] = v
        matrix_ad_[2][2] = 1.0
        matrix_ad_[2][3] = dt

        # b = [0.0, 0.0, 0.0, v / L].T
        matrix_bd_ = np.zeros((state_size, 1))  # time discrete b matrix
        matrix_bd_[3][0] = v / wheelbase

        return matrix_ad_, matrix_bd_


class LonController:
    """
    Longitudinal Controller using PID.
    """

    @staticmethod
    def ComputeControlCommand(target_speed, vehicle_state, dist):
        """
        calc acceleration command using PID.
        :param target_speed: target speed [m / s]
        :param vehicle_state: vehicle state
        :param dist: distance to goal [m]
        :return: control command (acceleration) [m / s^2]
        """

        a = 0.3 * (target_speed - vehicle_state.speed)

        if dist < 10.0:
            if vehicle_state.speed > 2.0:
                a = -3.0
            elif vehicle_state.speed < -2:
                a = -1.0

        return a


def main():

    import sys

    sys.path.append("./")

    from myterial import purple, indigo_dark, salmon_dark

    import draw
    import control
    from geometry import GrowingPath
    import geometry.vector_analysis as va

    from fcutils.progress import track

    # get path
    wps = control.paths.Waypoints(use="spline")
    path = control.paths.BSpline(wps.x, wps.y, degree=3)

    # draw path and waypoints
    f, ax = plt.subplots(figsize=(6, 10))
    draw.Hairpin()
    draw.Arrows(wps.x, wps.y, wps.theta, L=2)
    draw.Tracking(path.x, path.y)

    # plt.plot(path.curvature)
    # draw.Arrows(path.x[::10], path.y[::10], path.theta[::10], L=2, color='red')
    # plt.show()
    # raise ValueError

    # initialize car
    maxTime = 10.0

    # initialize controllers
    lat_controller = LatController()
    lon_controller = LonController()

    # initialize traj analyzer and vehicle
    ref_trajectory = TrajectoryAnalyzer(
        path.x, path.y, path.theta, path.curvature
    )
    vehicle = Vehicle(
        x=path.x[0], y=path.y[0], theta=path.theta[0], speed=path.speed[0],
    )

    trajectory = GrowingPath()
    for t in track(np.arange(0, maxTime, dt)):
        # compute error
        dist = math.hypot(vehicle.x - path.x[-1], vehicle.y - path.y[-1])
        target_speed = 25.0 / 3.6
        if dist <= 0.5:
            break

        # use controllers
        delta_opt, theta_e, e_cg = lat_controller.ComputeControlCommand(
            vehicle, ref_trajectory
        )
        a_opt = lon_controller.ComputeControlCommand(
            target_speed, vehicle, dist
        )

        # update vehicle
        vehicle.update_state(delta_opt, a_opt, e_cg, theta_e)

        # store trajectory
        trajectory.update(vehicle.x, vehicle.y, np.degrees(vehicle.theta))

        plt.cla()
        # draw track
        draw.Tracking(path.x, path.y, lw=3)

        # draw car trajectory
        draw.Tracking(trajectory.x, trajectory.y, lw=1.5, color=salmon_dark)
        draw.ControlCar(
            vehicle.x, vehicle.y, vehicle.theta, -vehicle.steer,
        )

        # draw car velocity and acceleration vectors
        if len(trajectory.x) > 5:
            velocity, _, _, acceleration, _, _ = va.compute_vectors(
                trajectory.x[:-3], trajectory.y[:-3]
            )

            draw.Arrow(
                vehicle.x,
                vehicle.y,
                velocity.angle[-1],
                L=velocity.magnitude[-1],
                color=indigo_dark,
            )
            draw.Arrow(
                vehicle.x,
                vehicle.y,
                acceleration.angle[-1],
                L=acceleration.magnitude[-1] * 10,
                color=purple,
            )

        plt.axis("equal")

        plt.title(
            "LQR (Kinematic): v=" + str(vehicle.speed * 3.6)[:4] + "km/h"
        )
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        plt.pause(0.001)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
