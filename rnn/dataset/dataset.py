import sys

from rnn.dataset._dataset import Preprocessing, Dataset


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
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history):
        return history.x, history.y, history.theta

    def get_outputs(self, history):
        return (
            history.tau_r,
            history.tau_l,
        )


class PredictNuDotFromXYT(Dataset, Preprocessing):
    description = """
        Predict wheel velocityies (nudot right/left) from 
        the  trajectory.

        The model predicts the wheels velocities, **not** the controls (taus)

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """

    name = "dataset_predict_nudot_from_xyt"
    inputs_names = ("x", "y", "theta")
    outputs_names = ("nudot_R", "nudot_L")

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history):
        return history.x, history.y, history.theta

    def get_outputs(self, history):
        return history.tau_r, history.tau_l


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
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history):
        return history.x, history.y, history.theta, history.v, history.omega

    def get_outputs(self, history):
        return history.tau_r, history.tau_l


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
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history):
        dx = history.goal_x - history.x
        dy = history.goal_y - history.y
        dtheta = history.goal_theta - history.theta

        return dx, dy, dtheta

    def get_outputs(self, history):
        return history.nudot_right, history.nudot_left


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
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history):
        dx = history.goal_x - history.x
        dy = history.goal_y - history.y
        dtheta = history.goal_theta - history.theta

        return dx, dy, dtheta

    def get_outputs(self, history):
        return history.tau_r, history.tau_l
