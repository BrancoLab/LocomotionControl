from numpy import sin, cos, sqrt, pi
import numpy as np

from PID.state import State
from PID import pid


class Robot:
    """
        Kinematics of a two wheels differential drive robot
    """

    R: float = 2  # cm | wheels radiuys
    L: float = 4  # cm | body width
    d: float = 3  # cm | distance between wheels axel and COM

    max_wheel_speed = 100

    def __init__(
        self, state: State, angle_pid: pid.PID, speed_pid: pid.PID, dt: float
    ):
        self.state = state
        self.angle_pid = angle_pid
        self.speed_pid = speed_pid
        self.dt = dt

        # initialize wheels angular velocities
        self.l_wheel_avel: float = 0  # rad/s
        self.r_wheel_avel: float = 0  # rad/s

    def rot_mtx(self):
        """
            Rotation matrix to map a point's cooridnates from the intertial
            frame to the robot's coordinates frame
        """
        theta = self.state.theta
        return np.array(
            [
                [cos(theta), -sin(theta), 0],
                [sin(theta), cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def update_speeds(self):
        """
            Based on the wheels instantaneous angular velocities
            update the robot's linear and angular velocity
        """
        self.state.speed = self.R / 2 * (self.l_wheel_avel + self.r_wheel_avel)
        self.state.angular_velocity = (
            self.R / 2 * (self.r_wheel_avel - self.l_wheel_avel)
        )

    def move(self):
        """
            It uses the robot's kinematics model to update the state (speeds and movements)
        """
        # get updated speed
        self.update_speeds()

        # get change in position and orientaiton
        theta = self.state.theta + pi / 2
        state_delta = State(
            cos(theta) * self.state.speed,  # x coord
            sin(theta) * self.state.speed,  # y coord
            self.state.angular_velocity,  # angle
        )

        # update state
        self.state = self.state + (state_delta * self.dt)

    def control(self, goal_state: State) -> float:
        """
            Uses PID to get the controls (change in speeds of wheels)
        """
        # compute errors
        angle_error = goal_state.theta - self.state.theta

        distance_error = sqrt(
            (goal_state.x - self.state.x) ** 2
            + (goal_state.y - self.state.y) ** 2
        )

        # compute controls
        angle_control = self.angle_pid(angle_error)
        speed_control = self.speed_pid(distance_error)

        # update wheels velocities
        self.l_wheel_avel += speed_control - angle_control
        self.r_wheel_avel += speed_control + angle_control

        # apply limits
        if self.l_wheel_avel < 0:
            self.l_wheel_avel = 0
        elif self.l_wheel_avel > self.max_wheel_speed:
            self.l_wheel_avel = self.max_wheel_speed

        if self.r_wheel_avel < 0:
            self.r_wheel_avel = 0
        elif self.r_wheel_avel > self.max_wheel_speed:
            self.r_wheel_avel = self.max_wheel_speed

        return angle_error, speed_control
