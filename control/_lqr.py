import numpy as np
from typing import Tuple
import math

from control.vehicle import Vehicle
from control.paths.utils import pi_2_pi


class TrajectoryAnalyzer:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        curvature: np.ndarray,
    ):
        self.x = x
        self.y = y
        self.theta = np.radians(theta)
        self.curvature = curvature

        # indices to keep track of relevant trajectory segmemt
        self.ind_old = 0
        self.ind_end = len(x)

    @property
    def horizon(self) -> int:
        return (
            self.ind_old + 20
            if self.ind_old + 20 < self.ind_end
            else self.ind_end
        )

    def to_trajectory_frame(self, vehicle: Vehicle) -> Tuple[float]:
        """
        errors to trajectory frame
        theta_error = theta_vehicle - theta_ref_path
        lateral_error = lateral distance of center of gravity (cg) in frenet frame
        :param vehicle: vehicle state (class Vehicle)
        :return: theta_error, lateral_error, theta_ref, curvature_ref
        """

        theta = vehicle.theta

        # calc nearest point in ref path
        dx = [vehicle.x - ix for ix in self.x[self.ind_old : self.ind_end]]
        dy = [vehicle.y - iy for iy in self.y[self.ind_old : self.ind_end]]

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
            lateral_error = 1.0 * dist  # vehicle on the right of ref path
        else:
            lateral_error = -1.0 * dist  # vehicle on the left of ref path

        # calc theta error: theta_error = theta_vehicle - theta_ref
        self.ind_old += ind_add
        theta_ref = self.theta[self.ind_old]
        theta_error = pi_2_pi(theta - theta_ref)

        # calc ref curvature
        curvature_ref = self.curvature[self.ind_old]

        return theta_error, lateral_error, theta_ref, curvature_ref


class LongitudinalController:
    """
    Longitudinal Controller using PID.
    """

    gain = 0.5

    def ComputeControlCommand(self, target_speed, vehicle_state, dist):
        """
        calc acceleration command using PID.
        :param target_speed: target speed [m / s]
        :param vehicle_state: vehicle state
        :param dist: distance to goal [m]
        :return: control command (acceleration) [m / s^2]
        """

        a = self.gain * (target_speed - vehicle_state.speed)

        if dist < 10.0:
            if vehicle_state.speed > 2.0:
                a = -3.0
            elif vehicle_state.speed < -2:
                a = -1.0

        return a
