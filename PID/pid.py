import numpy as np


class PID:
    def __init__(
        self,
        proportional_gain: float,
        integral_gain: float,
        derivative_gain: float,
        dt: float,
    ):
        # hyperparameters
        self.proportional_gain = proportional_gain
        self.integral_gain = integral_gain
        self.derivative_gain = derivative_gain
        self.dt = dt

        self.errors_history = []

    def __call__(self, error: float) -> float:
        """
            Computes the PID error
        """

        # compute error
        self.errors_history.append(error)

        # compute proportional, integral and derivative errors
        proportional = self.proportional_gain * error
        integral = self.integral_gain * np.sum(self.errors_history) / self.dt

        if len(self.errors_history) > 2:
            derivative = (
                self.derivative_gain
                * (error - self.errors_history[-2])
                / self.dt
            )
        else:
            derivative = 0

        return proportional + integral + derivative
