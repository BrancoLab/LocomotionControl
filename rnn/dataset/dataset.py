import numpy as np
import sys
from fcutils.maths.utils import rolling_mean


from rnn.dataset._dataset import Preprocessing, Dataset

# from proj.utils.misc import trajectory_at_each_simulation_step

is_win = sys.platform == "win32"


class PredictTauFromXYT(Dataset, Preprocessing):
    description = """
        Predict controls (tau right/left) from 
        the  trajectory XYT trajectory at each frame

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """

    name = "dataset_predict_tau_from_deltaXYT"
    inputs_names = ("x", "y", "theta")
    outputs_names = ("tau_R", "tau_L")

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history, window=21):
        x = rolling_mean(history.x, window)
        y = rolling_mean(history.y, window)
        theta = rolling_mean(history.theta, window)

        return x, y, theta

    def get_outputs(self, history, window=21):
        return (
            rolling_mean(history["tau_r"], window),
            rolling_mean(history["tau_l"], window),
        )


class PredictTauFromXYTVO(Dataset, Preprocessing):
    description = """
        Predict controls (tau right/left) from 
        the  trajectory XYTVO trajectory at each frame

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """

    name = "dataset_predict_tau_from_deltaXYTVO"
    inputs_names = ("x", "y", "theta", "v", "omega")
    outputs_names = ("tau_R", "tau_L")

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history, window=21):
        x = rolling_mean(history.x, window)
        y = rolling_mean(history.y, window)
        theta = rolling_mean(history.theta, window)
        v = rolling_mean(history.v, window)
        omega = rolling_mean(history.omega, window)

        return x, y, theta, v, omega

    def get_outputs(self, history, window=21):
        return (
            rolling_mean(history["tau_r"], window),
            rolling_mean(history["tau_l"], window),
        )


class PredictNudotFromDeltaXYT(Dataset, Preprocessing):
    description = """
        Predict wheel velocityies (nudot right/left) from 
        the  trajectory delta: difference betwen current state
        and next goal in the control simulation.

        The model predicts the wheels velocities, **not** the controls (taus)

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """

    name = "dataset_predict_nudot_from_deltaXYT"
    inputs_names = ("x", "y", "theta")
    outputs_names = ("nudot_R", "nudot_L")

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history, window=21):
        dx = rolling_mean(history.goal_x - history.x, window)
        dy = rolling_mean(history.goal_y - history.y, window)
        dtheta = rolling_mean(history.goal_theta - history.theta, window)

        return dx, dy, dtheta

    def get_outputs(self, history):
        return (
            np.array(history["nudot_right"]),
            np.array(history["nudot_left"]),
        )


class PredictTauFromDeltaXYT(Dataset, Preprocessing):
    description = """
        Predict controls (tau right/left) from 
        the  trajectory delta: difference betwen current state
        and next goal in the control simulation.

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """

    name = "dataset_predict_tau_from_deltaXYT"
    inputs_names = ("x", "y", "theta")
    outputs_names = ("tau_R", "tau_L")

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history, window=21):
        dx = rolling_mean(history.goal_x - history.x, window)
        dy = rolling_mean(history.goal_y - history.y, window)
        dtheta = rolling_mean(history.goal_theta - history.theta, window)

        return dx, dy, dtheta

    def get_outputs(self, history, window=21):
        return (
            rolling_mean(history["tau_r"], window),
            rolling_mean(history["tau_l"], window),
        )
