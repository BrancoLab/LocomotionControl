import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import medfilt
from rich.progress import track
from rich.prompt import Confirm
from rich import print
import numpy as np
import joblib

from pyinspect.utils import subdirs

from proj.rnn._utils import RNNLog
from proj.utils.misc import (
    load_results_from_folder,
    trajectory_at_each_simulation_step,
)


def get_delta_traj(trajectory, history):
    traj_sim = trajectory_at_each_simulation_step(trajectory, history)
    # goal_traj = history[
    #     ["goal_x", "goal_y", "goal_theta", "goal_v", "goal_omega"]
    # ].values

    # delta_traj = goal_traj[1:, :] - traj_sim[:-1, :]
    delta_traj = traj_sim[1:]  # ! Not using delta trajectory

    smoothed = np.zeros_like(delta_traj)
    for i in range(delta_traj.shape[1]):
        smoothed[:, i] = medfilt(delta_traj[:, i], 5)

    return smoothed[:-50, :]  # ! skipping the last few


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


class DatasetMaker(RNNLog):
    def __init__(self, trim_controls=50000):
        RNNLog.__init__(self, mk_dir=False)
        self.trim_controls = trim_controls

    def _standardize_dataset(self, trials_folders):
        print("Normalizing dataset")
        # Get normalizer
        if self.config["dataset_normalizer"] == "scale":
            input_scaler = MinMaxScaler(feature_range=(-1, 1))
            output_scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            input_scaler = StandardScaler()
            output_scaler = StandardScaler()

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
            delta_traj = get_delta_traj(trajectory, history)
            all_trajs.append(delta_traj)

            # stack outputs
            all_outputs.append(
                np.vstack([history["tau_r"][1:], history["tau_l"][1:]])
            )

        # fit normalizer
        _in, _out = np.vstack(all_trajs), np.hstack(all_outputs).T
        _out[_out > self.trim_controls] = self.trim_controls
        _out[_out < -self.trim_controls] = -self.trim_controls

        input_scaler = input_scaler.fit(_in)
        output_scaler = output_scaler.fit(_out)

        if self.config["interactive"]:
            print("Visualizing dataset")
            plot_dataset(_in, _out)

            if Confirm.ask("Continue with [b]dataset creation?", default=True):
                # Save normalizer to invert the process in the future
                joblib.dump(input_scaler, str(self.input_scaler_path))
                joblib.dump(output_scaler, str(self.output_scaler_path))

                return input_scaler, output_scaler
            else:
                print("Did not create dataset")
                return None, None
        else:
            joblib.dump(input_scaler, str(self.input_scaler_path))
            joblib.dump(output_scaler, str(self.output_scaler_path))

            return input_scaler, output_scaler

    def make_dataset(self):
        trials_folders = subdirs(self.trials_folder)

        input_scaler, output_scaler = self._standardize_dataset(trials_folders)
        if input_scaler is None:
            return

        print("Creating training data")
        data = dict(trajectory=[], tau_r=[], tau_l=[], sim_dt=[],)
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

            # Get trajectory point at each simulation step
            delta_traj = get_delta_traj(trajectory, history)
            norm_input = input_scaler.transform(delta_traj)

            out = np.vstack([history["tau_r"][:-1], history["tau_l"][:-1]]).T
            out[out > self.trim_controls] = self.trim_controls
            out[out < -self.trim_controls] = -self.trim_controls
            norm_output = output_scaler.transform(out)

            if self.config["dataset_normalizer"] == "scale":
                # ? When scaling ignore small trials
                if (
                    np.max(norm_output[50:-50, :]) < 0.2
                    and np.min(norm_output[50:-50, :]) > -0.2
                ):
                    continue

            # Append to dataset
            data["trajectory"].append(norm_input)
            data["tau_r"].append(norm_output[:, 0])
            data["tau_l"].append(norm_output[:, 1])
            data["sim_dt"].append(config["dt"])

        # Save data to file
        if self.config["interactive"]:
            plot_dataset(
                np.vstack(data["trajectory"]),
                np.vstack(
                    [
                        np.concatenate(data["tau_r"]),
                        np.concatenate(data["tau_l"]),
                    ]
                ).T,
            )

            if Confirm.ask("Save dataset?", default=True):
                pd.DataFrame(data).to_hdf(self.dataset_path, key="hdf")
                self._data = pd.read_hdf(self.dataset_path, key="hdf")
                self._n_trials = len(data["sim_dt"])
                print(f"Saved at {self.dataset_path}, {self._n_trials} trials")
            else:
                print("Did not save dataset")
        else:
            pd.DataFrame(data).to_hdf(self.dataset_path, key="hdf")
            self._data = pd.read_hdf(self.dataset_path, key="hdf")
            self._n_trials = len(data["sim_dt"])
            print(f"Saved at {self.dataset_path}, {self._n_trials} trials")
