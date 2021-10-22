import math
import numpy as np
from typing import Tuple

from control.config import (
    dt,
    wheelbase,
    max_steer_angle,
    look_ahead_distance,
    look_ahead_gain,
)
from control.vehicle import Vehicle
from control.paths.utils import pi_2_pi


class Node:
    def __init__(self, x, y, theta, speed):
        self.x = x
        self.y = y
        self.theta = np.radians(theta)
        self.speed = speed

    def update(
        self, acceleration, steering,
    ):
        # steering = self.limit_input(steering)
        self.x += self.speed * math.cos(self.theta) * dt
        self.y += self.speed * math.sin(self.theta) * dt
        self.theta += self.speed / wheelbase * math.tan(steering) * dt
        self.speed += acceleration * dt

    @staticmethod
    def limit_input(delta):
        if delta > 1.2 * max_steer_angle:
            return 1.2 * max_steer_angle

        if delta < -1.2 * max_steer_angle:
            return -1.2 * max_steer_angle

        return delta


class Nodes:
    def __init__(self):
        self.x = []
        self.y = []
        self.theta = []
        self.speed = []
        self.t = []

    def add(self, t, node):
        self.x.append(node.x)
        self.y.append(node.y)
        self.theta.append(node.theta)
        self.speed.append(node.speed)
        self.t.append(t)


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
        self.ind_end = len(x) - 1

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

    def target_index(self, node):
        """
        search ind of target point in the reference path.
        the distance between target point and current position is ld
        :param node: current information
        :return: ind of target point
        """

        if self.ind_old is None:
            self.calc_nearest_ind(node)

        Lf = look_ahead_gain * node.speed + look_ahead_distance

        for ind in range(self.ind_old, self.ind_end + 1):
            if self.calc_distance(node, ind) > Lf:
                self.ind_old = ind
                return ind, Lf

        self.ind_old = self.ind_end

        return self.ind_end, Lf

    def calc_nearest_ind(self, node):
        """
        calc ind of the nearest point to current position
        :param node: current information
        :return: ind of nearest point
        """

        dx = [node.x - x for x in self.x]
        dy = [node.y - y for y in self.y]
        ind = np.argmin(np.hypot(dx, dy))
        self.ind_old = ind

    def calc_distance(self, node, ind):
        return math.hypot(node.x - self.x[ind], node.y - self.y[ind])
