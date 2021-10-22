"""
    Code adapted from: https://github.com/zhm-real/MotionPlanning
    zhm-real shared code to create different types of paths under an MIT license.
    The logic of the code is left un-affected here, I've just refactored it.
"""

import sys
import math
import numpy as np

sys.path.append("./")

from geometry import Path, GrowingPath

from control.config import (
    wheelbase,
    dt,
    max_iteration,
    eps,
    matrix_q,
    matrix_r,
    state_size,
)
from control.vehicle import Vehicle
from control._control import LongitudinalController
from control.trajectory import TrajectoryAnalyzer


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

        lateral_error_old = vehicle_state.lateral_error
        theta_error_old = vehicle_state.theta_error

        (
            theta_error,
            lateral_error,
            theta_ref,
            curvature_reference,
        ) = ref_trajectory.to_trajectory_frame(vehicle_state)

        matrix_ad_, matrix_bd_ = self.UpdateMatrix(vehicle_state)

        matrix_state_ = np.zeros((state_size, 1))
        matrix_r_ = np.diag(matrix_r)
        matrix_q_ = np.diag(matrix_q)

        matrix_k_ = self.SolveLQRProblem(
            matrix_ad_, matrix_bd_, matrix_q_, matrix_r_, eps, max_iteration
        )

        matrix_state_[0][0] = lateral_error
        matrix_state_[1][0] = (lateral_error - lateral_error_old) / dt
        matrix_state_[2][0] = theta_error
        matrix_state_[3][0] = (theta_error - theta_error_old) / dt

        steer_angle_feedback = -(matrix_k_ @ matrix_state_)[0][0]

        steer_angle_feedforward = self.ComputeFeedForward(curvature_reference)

        steer_angle = steer_angle_feedback + steer_angle_feedforward

        return steer_angle, theta_error, lateral_error

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


class KinematicsLQR:
    def __init__(self, path: Path):
        self.path = path

        # initialize controllers
        self.lat_controller = LatController()
        self.lon_controller = LongitudinalController()

        # initialize traj analyzer
        self.target_trajectory = TrajectoryAnalyzer(
            path.x, path.y, path.theta, path.curvature
        )
        self.trajectory = GrowingPath()  # store simulated trajectory

        # initialize vehicle
        self.vehicle = Vehicle(
            x=path.x[0], y=path.y[0], theta=path.theta[0], speed=path.speed[0],
        )

    def step(self, t: FloatingPointError) -> float:
        """
            Solves planning and LQR and steps the vehicle
        """
        # compute error
        dist = math.hypot(
            self.vehicle.x - self.path.x[-1], self.vehicle.y - self.path.y[-1]
        )
        target_speed = self.path.speed[self.target_trajectory.ind_old]

        # use controllers
        (
            steering,
            theta_error,
            lateral_error,
        ) = self.lat_controller.ComputeControlCommand(
            self.vehicle, self.target_trajectory
        )
        acceleration = self.lon_controller.ComputeControlCommand(
            target_speed, self.vehicle, dist
        )

        # update vehicle
        self.vehicle.update_state(
            steering, acceleration, lateral_error, theta_error
        )

        # store trajectory
        self.trajectory.update(
            self.vehicle.x, self.vehicle.y, np.degrees(self.vehicle.theta)
        )

        return target_speed


#  TODO  add gains to controllers
# TODO add dynamics model
