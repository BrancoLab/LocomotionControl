import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from rich.progress import track
from rich import print
import numpy as np


from pyinspect.utils import subdirs

from proj.rnn._utils import RNNPaths
from proj.utils.misc import (
    load_results_from_folder,
    trajectory_at_each_simulation_step,
)


"""
    Preprocess results from running the control
    algorithm on a number of trials to create a 
    normalized dataset for using with RNNs.
"""

# ------------------------- old inputs outputs funcs ------------------------- #

# def get_inputs(trajectory, history):
#     traj_sim = trajectory_at_each_simulation_step(trajectory, history)
#     # goal_traj = history[
#     #     ["goal_x", "goal_y", "goal_theta", "goal_v", "goal_omega"]
#     # ].values

#     # delta_traj = goal_traj - traj_sim
#     return traj_sim


# def get_outputs(history):
#     # return np.vstack([history["tau_r"][1:], history["tau_l"][1:]])
#     return np.vstack([history["nudot_right"], history["nudot_left"]])


# ------------------------- base pre-processing class ------------------------ #


class Preprocessing(RNNPaths):
    """
        Class to take the results of iLQR and organize them
        into a structured dataset that can be used for training RNNs
    """

    name = "dataset name"
    description = "base"  # updated in subclasses to describe dataset

    # names of inputs and outputs of dataset
    _data = (("x", "y", "theta", "v", "omega"), ("out1", "out2"))

    def __init__(self):
        RNNPaths.__init__(self, dataset_name=self.name)

    def get_inputs(self, trajectory, history):
        return NotImplementedError(
            "get_inputs should be implemented in your dataset preprocessing"
        )
        # should return x,y,theta,v,omega

    def get_outputs(self, history):
        return NotImplementedError(
            "get_outputs should be implemented in your dataset preprocessing"
        )
        # should return output1, output2

    def fit_scaler(self, df):
        # concatenate the values under each columns to fit a scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data = pd.DataFrame(
            {c: np.concatenate(df[c].values) for c in df.columns}
        )
        return scaler.fit(data)

    def scale(self, df, scaler):
        """
            Use a fitted minmax scaler to scale
            each trial in a dataframe
        """
        scaled = {c: [] for c in df.columns}
        for i, t in df.iterrows():
            scld = scaler.transform(np.vstack(t.values).T)
            for n, c in enumerate(df.columns):
                scaled[c].append(scld[:, n])

        return pd.DataFrame(scaled)

    def split_and_normalize(self, data):
        train, test = train_test_split(data)

        train_scaler = self.fit_scaler(train)
        train = self.scale(train, train_scaler)

        test_scaler = self.fit_scaler(train)
        test = self.scale(test, test_scaler)

        self.save_normalizers(train_scaler, test_scaler)
        return train, test

    def describe(self):
        with open(self.dataset_folder / "description.txt", "w") as out:
            out.write(self.description)

    def make(self):
        """ 
            Organizes the standardized data into a single dataframe.
        """
        trials_folders = subdirs(self.trials_folder)
        print(
            f"[bold magenta]Creating dataset...\nFound {len(trials_folders)} trials folders."
        )

        # Create dataframe with all trials
        data = {
            **{k: [] for k in self._data[0]},
            **{k: [] for k in self._data[1]},
        }
        for fld in track(trials_folders):
            try:
                (
                    config,
                    trajectory,
                    history,
                    cost_history,
                    trial,
                    info,
                ) = load_results_from_folder(fld)
            except Exception:
                print(f"Could not open a trial folder, skipping [{fld.name}]")
                continue

            # Get inputs
            inputs = self.get_inputs(trajectory, history)
            for name, value in zip(self._data[0], inputs):
                data[name].append(value)

            # get outputs
            outputs = self.get_outputs(history)
            for name, value in zip(self._data[1], outputs):
                data[name].append(value)

        # as dataframe
        data = pd.DataFrame(data)

        # split and normalize
        train, test = self.split_and_normalize(data)

        # save
        self.save_dataset(train, test)
        self.describe()


# ---------------------------------------------------------------------------- #
#                             preprocessing classes                            #
# ---------------------------------------------------------------------------- #


class PredictNudotFromXYT(Preprocessing):
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

    def __init__(self):
        Preprocessing.__init__(self)

    def get_inputs(self, trajectory, history):
        trj = trajectory_at_each_simulation_step(trajectory, history)
        x, y, theta = trj[:, 0], trj[:, 1], trj[:, 2]
        return x, y, theta

    def get_outputs(self, history):
        return (
            np.array(history["nudot_right"]),
            np.array(history["nudot_left"]),
        )
