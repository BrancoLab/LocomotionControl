from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    x: float
    y: float
    theta: float  # orientation
    speed: float = 0
    angular_velocity: float = 0

    def __sub__(self, other) -> float:
        """
            Subtracts the XYTHETA parameters betwen two states
        """

        return np.sum(
            [
                # self.x - other.x,
                # self.y - other.y,
                self.theta
                - other.theta,
            ]
        )

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
        self.theta += other.theta
        self.speed += other.speed
        self.angular_velocity += other.angular_velocity

        return self

    def __mul__(self, value: float):
        self.x *= value
        self.y *= value
        self.theta *= value
        self.speed *= value
        self.angular_velocity *= value

        return self
