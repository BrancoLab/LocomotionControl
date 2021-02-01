import sys

sys.path.append("./")
from rnn.dataset._dataset import Preprocessing, Dataset


is_win = sys.platform == "win32"


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


if __name__ == "__main__":
    datasets = (PredictTauFromDeltaXYT,)

    for dataset in datasets:
        dataset().make()
        dataset().plot_random()
        dataset().plot_durations()
