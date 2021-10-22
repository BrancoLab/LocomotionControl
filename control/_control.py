import numpy as np


from fcutils.maths import derivative


from control.config import (
    longitudinal_pid_P_gain,
    longitudinal_pid_D_gain,
    longitudinal_pid_I_gain,
)

"""
    Trajectory analyzer and longitudinal PID controller for LQR based
    control scripts
"""


class LongitudinalController:
    """
    Longitudinal Controller using PID.
    """

    def __init__(self):
        self.error_history = []

    def __call__(self, target_speed, speed, dist):
        a = longitudinal_pid_P_gain * (target_speed - speed)

        if dist < 10.0:
            if speed > 3.0:
                a = -2.5
            elif speed < -2.0:
                a = -1.0

        return a

    def ComputeControlCommand(self, target_speed, vehicle_state, dist):
        """
        calc acceleration command using PID.
        :param target_speed: target speed [m / s]
        :param vehicle_state: vehicle state
        :param dist: distance to goal [m]
        :return: control command (acceleration) [m / s^2]
        """
        error = target_speed - vehicle_state.speed
        self.error_history.append(error)

        a = 0
        a += longitudinal_pid_P_gain * error
        a += longitudinal_pid_D_gain * derivative(self.error_history)[-1]

        if len(self.error_history) > 6:
            a += longitudinal_pid_I_gain * np.sum(self.error_history[-5:])

        if dist < 10.0:
            if vehicle_state.speed > 2.0:
                a = -3.0
            elif vehicle_state.speed < -2:
                a = -1.0

        return a
