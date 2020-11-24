import numpy as np
import matplotlib.pyplot as plt
from myterial import (
    salmon,
    salmon_dark,
    light_green,
    light_green_dark,
)
import sys
from pyrnn._plot import clean_axes


from proj.rnn._dataset import Preprocessing, Dataset
from proj.utils.misc import trajectory_at_each_simulation_step

is_win = sys.platform == "win32"


class PredictNudotFromXYT(Dataset, Preprocessing):
    description = """
        Predict wheel velocityies (nudot right/left) from 
        the 'trajectory at each step' (i.e. the next trajectory waypoint
        at each frame in the simulation, to match the inputs 
        and controls produced by the control model).

        The model predicts the wheels velocities, **not** the controls (taus)

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """

    name = "dataset_predict_nudot_from_XYT"
    _data = (("x", "y", "theta"), ("nudot_R", "nudot_L"))

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history):
        trj = trajectory_at_each_simulation_step(trajectory, history)
        x, y, theta = trj[:, 0], trj[:, 1], trj[:, 2]
        return x, y, theta

    def get_outputs(self, history):
        return (
            np.array(history["nudot_right"]),
            np.array(history["nudot_left"]),
        )


class PredictNudotFromDeltaXYT(Dataset, Preprocessing):
    description = """
        Predict wheel velocityies (nudot right/left) from 
        the 'trajectory at each step' (i.e. the next trajectory waypoint
        at each frame in the simulation, to match the inputs 
        and controls produced by the control model).

        The model predicts the wheels velocities, **not** the controls (taus)

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """

    name = "dataset_predict_nudot_from_deltaXYT"
    _data = (("x", "y", "theta"), ("nudot_R", "nudot_L"))

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history):
        trj = trajectory_at_each_simulation_step(trajectory, history)
        x, y, theta = trj[:, 0], trj[:, 1], trj[:, 2]

        gtraj = history[["goal_x", "goal_y", "goal_theta"]].values

        return x - gtraj[:, 0], y - gtraj[:, 1], theta - gtraj[:, 2]

    def get_outputs(self, history):
        return (
            np.array(history["nudot_right"]),
            np.array(history["nudot_left"]),
        )


def plot_predictions(model, batch_size, dataset, **kwargs):
    """
    Run the model on a single batch and plot
    the model's prediction's against the
    input data and labels.
    """
    X, Y = dataset.get_one_batch(1, **kwargs)

    if model.on_gpu:
        model.cpu()
        model.on_gpu = False

    o, h = model.predict(X)

    n_inputs = X.shape[-1]
    n_outputs = Y.shape[-1]
    labels = ["x", "y", "$\\theta$", "v", "$\\omega$"]

    f, axarr = plt.subplots(nrows=2, figsize=(12, 9))

    for n in range(n_inputs):
        axarr[0].plot(X[0, :, n], lw=2, label=labels[n])
    axarr[0].set(title="inputs")
    axarr[0].legend()

    cc = [salmon, light_green]
    oc = [salmon_dark, light_green_dark]
    labels = ["nudot_R", "nudot_L"]
    for n in range(n_outputs):
        axarr[1].plot(
            Y[0, :, n], lw=2, color=cc[n], label="correct " + labels[n]
        )
        axarr[1].plot(
            o[0, :, n], lw=2, ls="--", color=oc[n], label="model output"
        )
    axarr[1].legend()
    axarr[1].set(title="outputs")

    f.tight_layout()
    clean_axes(f)
