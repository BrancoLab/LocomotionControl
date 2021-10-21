from typing import Tuple
import numpy as np
import math

from control.config import (
    wheelbase,
    dt,
    max_acceleration,
    max_steer_angle,
    max_speed,
)


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
