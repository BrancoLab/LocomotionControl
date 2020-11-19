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

    description = "base"  # updated in subclasses to describe dataset

    def __init__(self):
        self.name = "preprocessing"
        RNNPaths.__init__(self, mk_dir=False)

    def get_inputs(self, trajectory, history):
        return NotImplementedError(
            "get_inputs should be implemented in your dataset preprocessing"
        )

    def get_outputs(self, history):
        return NotImplementedError(
            "get_outputs should be implemented in your dataset preprocessing"
        )

    def _standardize_dataset(self, trials_folders):
        """ 
            Creates preprocessing tools to standardize the dataset's
            inputs and outputs to facilitate RNN training.
            It pools data for the whole dataset to fit the standardizers.
        """
        print("Normalizing dataset")

        # Get normalizer
        input_scaler = MinMaxScaler(feature_range=(-1, 1))
        output_scaler = MinMaxScaler(feature_range=(-1, 1))

        # Get ALL data to fit the normalizer
        all_trajs, all_outputs = [], []
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
                continue

            if info["traj_duration"] < 1:
                continue

            # stack inputs
            delta_traj = self.get_inputs(trajectory, history)
            all_trajs.append(delta_traj)

            # stack outputs
            all_outputs.append(self.get_outputs(history))

        # fit normalizer
        _in, _out = np.vstack(all_trajs), np.hstack(all_outputs).T

        input_scaler = input_scaler.fit(_in)
        output_scaler = output_scaler.fit(_out)

        self.save_normalizers(input_scaler, output_scaler)
        return input_scaler, output_scaler

    def _split_and_save(self, data):
        """ 
            Splits the dataset into training and test data
            before saving.
        """
        data = pd.DataFrame(data)
        train, test = train_test_split(data)

        train.to_hdf(self.dataset_train_path, key="hdf")
        test.to_hdf(self.dataset_test_path, key="hdf")

        print(f"Saved at {self.dataset_train_path}, {len(data)} trials")

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

        input_scaler, output_scaler = self._standardize_dataset(trials_folders)

        print("Creating training data")
        data = dict(trajectory=[], controls=[], sim_dt=[],)
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
                continue

            # Get inputs and outputs
            delta_traj = self.get_inputs(trajectory, history)
            out = self.get_outputs(history).T

            if len(delta_traj) != len(out):
                raise ValueError(f"Length of input and outputs doesnt match")

            # Get stard and end
            start_idx = history.trajectory_idx.values[0]
            end_idx = history.trajectory_idx.values[-1] - 30

            # Normalize data
            norm_input = input_scaler.transform(
                delta_traj[start_idx:end_idx, :]
            )
            norm_output = output_scaler.transform(out[start_idx:end_idx, :])

            if len(norm_input) != len(norm_output):
                raise ValueError(f"Length of input and outputs doesnt match")

            # Append to dataset
            data["trajectory"].append(norm_input)
            data["controls"].append(norm_output)
            data["sim_dt"].append(config["dt"])

        # Save data to file
        self._split_and_save(data)


# ---------------------------------------------------------------------------- #
#                             preprocessing classes                            #
# ---------------------------------------------------------------------------- #


class PredictNudotPreProcessing(Preprocessing):
    description = """
        Predict wheel velocityies (nudot right/left) from 
        the 'trajectory at each step' (i.e. the next trajectory waypoint
        at each frame in the simulation, to match the inputs 
        and controls produced by the control model).

        The model predicts the wheels velocities, **not** the controls (taus)

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """

    def __init__(self):
        Preprocessing.__init__(self)

    def get_inputs(self, trajectory, history):
        return trajectory_at_each_simulation_step(trajectory, history)

    def get_outputs(self, history):
        return np.vstack([history["nudot_right"], history["nudot_left"]])
