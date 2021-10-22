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

from control.config import wheelbase
from control.vehicle import Vehicle
from control._control import LongitudinalController
from control.trajectory import TrajectoryAnalyzer, Node, Nodes


def pure_pursuit(node, ref_path, index_old):
    """
    pure pursuit controller
    :param node: current information
    :param ref_path: reference path: x, y, yaw, curvature
    :param index_old: target index of last time
    :return: optimal steering angle
    """

    ind, Lf = ref_path.target_index(node)  # target point and pursuit distance
    ind = max(ind, index_old)

    tx = ref_path.x[ind]
    ty = ref_path.y[ind]

    alpha = math.atan2(ty - node.y, tx - node.x) - node.yaw
    delta = math.atan2(2.0 * wheelbase * math.sin(alpha), Lf)

    return delta, ind


class PurePursuit:
    def __init__(self, path: Path):
        self.path = path
        self.trajectory = GrowingPath()
        self.target_trajectory = TrajectoryAnalyzer(
            path.x, path.y, path.theta, path.curvature
        )

        self.node = Node(
            x=path.x[0], y=path.y[0], theta=path.theta[0], speed=0.0
        )
        self.nodes = Nodes()
        self.nodes.add(0, self.node)

        self.pid_control = LongitudinalController()
        self.vehicle = Vehicle(
            x=path.x[0], y=path.y[0], theta=path.theta[0], speed=path.speed[0]
        )

    def step(self, t: float) -> float:
        """
                Solves planning and steps the vehicle
            """
        target_ind, _ = self.target_trajectory.target_index(self.node)
        target_speed = self.path.speed[self.target_trajectory.ind_old]

        xt = self.node.x + 1.1 * math.cos(self.node.theta)
        yt = self.node.y + 1.1 * math.sin(self.node.theta)
        dist = math.hypot(xt - self.path.x[-1], yt - self.path.y[-1])

        acceleration = self.pid_control(target_speed, self.node.speed, dist)
        steering, target_ind = self.pure_pursuit(
            self.node, self.target_trajectory, target_ind
        )

        self.node.update(acceleration, steering)
        self.nodes.add(t, self.node)

        self.vehicle.update_state(steering, acceleration, 0, 0)
        self.trajectory.update(
            self.vehicle.x, self.vehicle.y, np.degrees(self.vehicle.theta)
        )

        return target_speed

    @staticmethod
    def pure_pursuit(node, ref_path, index_old):
        """
        pure pursuit controller
        :param node: current information
        :param ref_path: reference path: x, y, yaw, curvature
        :param index_old: target index of last time
        :return: optimal steering angle
        """

        ind, Lf = ref_path.target_index(
            node
        )  # target point and pursuit distance
        ind = max(ind, index_old)

        tx = ref_path.x[ind]
        ty = ref_path.y[ind]

        alpha = math.atan2(ty - node.y, tx - node.x) - node.theta
        delta = math.atan2(2.0 * wheelbase * math.sin(alpha), Lf)

        return delta, ind
