"""
LQR and PID Controller
author: huiming zhou
"""

import sys
import math
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

sys.path.append("./")

from control.config import (
    wheelbase,
    l_f,
    l_r,
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


class Gear(Enum):
    GEAR_DRIVE = 1
    GEAR_REVERSE = 2


class VehicleState:
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, gear=Gear.GEAR_DRIVE):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.e_cg = 0.0
        self.theta_e = 0.0

        self.gear = gear
        self.steer = 0.0

    def UpdateVehicleState(
        self, delta, a, e_cg, theta_e, gear=Gear.GEAR_DRIVE
    ):
        """
        update states of vehicle
        :param theta_e: theta error to ref trajectory
        :param e_cg: lateral error to ref trajectory
        :param delta: steering angle [rad]
        :param a: acceleration [m / s^2]
        :param gear: gear mode [GEAR_DRIVE / GEAR/REVERSE]
        """

        wheelbase_ = l_r + l_f
        delta, a = self.RegulateInput(delta, a)

        self.gear = gear
        self.steer = delta
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.v / wheelbase_ * math.tan(delta) * dt
        self.e_cg = e_cg
        self.theta_e = theta_e

        if gear == Gear.GEAR_DRIVE:
            self.v += a * dt
        else:
            self.v += -1.0 * a * dt

        self.v = self.RegulateOutput(self.v)

    @staticmethod
    def RegulateInput(delta, a):
        """
        regulate delta to : - max_steer_angle ~ max_steer_angle
        regulate a to : - max_acceleration ~ max_acceleration
        :param delta: steering angle [rad]
        :param a: acceleration [m / s^2]
        :return: regulated delta and acceleration
        """

        if delta < -1.0 * max_steer_angle:
            delta = -1.0 * max_steer_angle

        if delta > 1.0 * max_steer_angle:
            delta = 1.0 * max_steer_angle

        if a < -1.0 * max_acceleration:
            a = -1.0 * max_acceleration

        if a > 1.0 * max_acceleration:
            a = 1.0 * max_acceleration

        return delta, a

    @staticmethod
    def RegulateOutput(v):
        """
        regulate v to : -max_speed ~ max_speed
        :param v: calculated speed [m / s]
        :return: regulated speed
        """

        max_speed_ = max_speed

        if v < -1.0 * max_speed_:
            v = -1.0 * max_speed_

        if v > 1.0 * max_speed_:
            v = 1.0 * max_speed_

        return v


class TrajectoryAnalyzer:
    def __init__(self, x, y, theta, k):
        self.x_ = x
        self.y_ = y
        self.theta_ = theta
        self.k_ = k

        self.ind_old = 0
        self.ind_end = len(x)

    def ToTrajectoryFrame(self, vehicle_state):
        """
        errors to trajectory frame
        theta_e = theta_vehicle - theta_ref_path
        e_cg = lateral distance of center of gravity (cg) in frenet frame
        :param vehicle_state: vehicle state (class VehicleState)
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

        ts_ = dt
        e_cg_old = vehicle_state.e_cg
        theta_e_old = vehicle_state.theta_e

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
        matrix_state_[1][0] = (e_cg - e_cg_old) / ts_
        matrix_state_[2][0] = theta_e
        matrix_state_[3][0] = (theta_e - theta_e_old) / ts_

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

        ts_ = dt
        wheelbase_ = l_f + l_r

        v = vehicle_state.v

        matrix_ad_ = np.zeros(
            (state_size, state_size)
        )  # time discrete A matrix

        matrix_ad_[0][0] = 1.0
        matrix_ad_[0][1] = ts_
        matrix_ad_[1][2] = v
        matrix_ad_[2][2] = 1.0
        matrix_ad_[2][3] = ts_

        # b = [0.0, 0.0, 0.0, v / L].T
        matrix_bd_ = np.zeros((state_size, 1))  # time discrete b matrix
        matrix_bd_[3][0] = v / wheelbase_

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

        if vehicle_state.gear == Gear.GEAR_DRIVE:
            direct = 1.0
        else:
            direct = -1.0

        a = 0.3 * (target_speed - direct * vehicle_state.v)

        if dist < 10.0:
            if vehicle_state.v > 2.0:
                a = -3.0
            elif vehicle_state.v < -2:
                a = -1.0

        return a


def main():

    import sys

    sys.path.append("./")

    import draw
    import control
    from geometry import GrowingPath

    from fcutils.progress import track

    # get path
    wps = control.paths.Waypoints()
    path = control.paths.DubinPath(wps).fit()

    # draw path and waypoints
    f, ax = plt.subplots(figsize=(6, 10))
    draw.Hairpin()
    draw.Arrows(wps.x, wps.y, wps.theta, L=2)
    draw.Tracking(path.x, path.y)

    # draw normal vector
    draw.Arrows(
        path.x[::10],
        path.y[::10],
        path.normal_angle[::10],
        L=2,
        color="salmon",
        label="normal",
    )

    # initialize car
    maxTime = 100.0

    # initialize controllers
    lat_controller = LatController()
    lon_controller = LonController()

    # initialize traj analyzer and vehicle
    ref_trajectory = TrajectoryAnalyzer(
        path.x, path.y, path.theta, path.curvature
    )
    vehicle_state = VehicleState(
        x=path.x[0], y=path.y[0], theta=path.theta[0], v=path.speed[0],
    )

    trajectory = GrowingPath()
    for t in track(np.arange(0, maxTime, dt)):
        # compute error
        dist = math.hypot(
            vehicle_state.x - path.x[-1], vehicle_state.y - path.y[-1]
        )
        target_speed = 25.0 / 3.6
        if dist <= 0.5:
            break

        # use controllers
        delta_opt, theta_e, e_cg = lat_controller.ComputeControlCommand(
            vehicle_state, ref_trajectory
        )
        a_opt = lon_controller.ComputeControlCommand(
            target_speed, vehicle_state, dist
        )

        # update vehicle
        vehicle_state.UpdateVehicleState(delta_opt, a_opt, e_cg, theta_e)

        # store trajectory
        trajectory.update(
            vehicle_state.x, vehicle_state.y, np.degrees(vehicle_state.theta)
        )

        plt.cla()
        draw.Tracking(path.x, path.y, lw=3)
        draw.Tracking(trajectory.x, trajectory.y, lw=1.5, color="salmon")
        draw.ControlCar(
            vehicle_state.x,
            vehicle_state.y,
            vehicle_state.theta,
            -vehicle_state.steer,
        )

        plt.axis("equal")

        plt.title(
            "LQR (Kinematic): v=" + str(vehicle_state.v * 3.6)[:4] + "km/h"
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
