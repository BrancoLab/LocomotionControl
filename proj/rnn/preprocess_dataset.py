import pandas as pd
import matplotlib.pyplot as plt
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


def get_inputs(trajectory, history):
    traj_sim = trajectory_at_each_simulation_step(trajectory, history)
    # goal_traj = history[
    #     ["goal_x", "goal_y", "goal_theta", "goal_v", "goal_omega"]
    # ].values

    # delta_traj = goal_traj - traj_sim
    return traj_sim


def get_outputs(history):
    # return np.vstack([history["tau_r"][1:], history["tau_l"][1:]])
    return np.vstack([history["nudot_right"], history["nudot_left"]])


def plot_dataset(inputs, outputs):
    f, axarr = plt.subplots(ncols=4, nrows=2, figsize=(20, 12))
    titles = [
        "X",
        "Y",
        "\\theta",
        "v",
        "\\omega",
        "\\tau_{R}",
        "\\tau_{L}",
    ]
    for n, (ax, ttl) in enumerate(zip(axarr.flatten(), titles)):
        if n <= 4:
            ax.hist(inputs[:, n], bins=100)
        else:
            ax.hist(outputs[:, n - 5], bins=100)
        ax.set(title=f"${ttl}$")
    plt.show()


class Preprocessing(RNNPaths):
    """
        Class to take the results of iLQR and organize them
        into a structured dataset that can be used for training RNNs
    """

    def __init__(self):
        self.name = "preprocessing"
        RNNPaths.__init__(self, mk_dir=False)

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
            delta_traj = get_inputs(trajectory, history)
            all_trajs.append(delta_traj)

            # stack outputs
            all_outputs.append(get_outputs(history))

        # fit normalizer
        _in, _out = np.vstack(all_trajs), np.hstack(all_outputs).T

        input_scaler = input_scaler.fit(_in)
        output_scaler = output_scaler.fit(_out)

        self.save_normalizers(input_scaler, output_scaler, _in, _out)
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

    def make_dataset(self):
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
            delta_traj = get_inputs(trajectory, history)
            out = get_outputs(history).T

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
