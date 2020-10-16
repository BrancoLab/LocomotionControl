from pyinspect.utils import subdirs
import numpy as np
from pathlib import Path
from rich import print
from rich.progress import track
from rich.prompt import Confirm
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from proj.paths import rnn_trainig
from proj.utils.misc import (
    load_results_from_folder,
    trajectory_at_each_simulation_step,
)
from proj.rnn.tasks.task import Task


class ControlTask(Task):
    def __init__(
        self,
        dt,
        tau,
        T,
        N_batch,
        n_inputs=5,
        n_outputs=2,
        data_path=None,
        trim_controls=50000,
    ):
        """
            Args:
                N_in (int): The number of network inputs.
                N_out (int): The number of network outputs.
                dt (float): The simulation timestep.
                tau (float): The intrinsic time constant of neural state decay.
                T (float): The trial length.
                N_batch (int): The number of trials per training update.

        """
        super(ControlTask, self).__init__(
            n_inputs, n_outputs, dt, tau, T, N_batch
        )
        self.trim_controls = trim_controls

        if data_path is None:
            self.data_path = rnn_trainig
        else:
            self.data_path = data_path

        self._folder = Path(self.data_path).parent
        self.dataset_path = self._folder / "training_data.h5"
        self.input_scaler_path = self._folder / "input_scaler.gz"
        self.output_scaler_path = self._folder / "output_scaler.gz"

        try:
            self._data = pd.read_hdf(self.dataset_path, key="hdf")
            self._n_trials = len(self._data)
        except FileNotFoundError:
            print("Did not find data file, make data?")
            self._make_data()

    @staticmethod
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
                ax.hist(inputs[:, n], bins=50)
            else:
                ax.hist(outputs[:, n - 5], bins=50)
            ax.set(title=f"${ttl}$")
        plt.show()

    def _standardize_dataset(self, trials_folders):
        print("Normalizing dataset")
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        output_scaler = MinMaxScaler(feature_range=(0, 1))

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
            traj_sim = trajectory_at_each_simulation_step(trajectory, history)
            delta_traj = traj_sim[1:, :] - traj_sim[:-1, :]
            all_trajs.append(delta_traj)

            # stack outputs
            all_outputs.append(
                np.vstack([history["tau_r"][1:], history["tau_l"][1:]])
            )

        # plot a histogram of all variables for inspection
        print("Visualizing dataset")
        _in, _out = np.vstack(all_trajs), np.hstack(all_outputs).T
        _out[_out > self.trim_controls] = self.trim_controls
        _out[_out < -self.trim_controls] = -self.trim_controls

        self.plot_dataset(_in, _out)

        if Confirm.ask("Continue with [b]dataset creation?", default=True):
            input_scaler = input_scaler.fit(_in)
            output_scaler = output_scaler.fit(_out)

            # Save normalizer to invert the process in the future
            joblib.dump(input_scaler, str(self.input_scaler_path))
            joblib.dump(output_scaler, str(self.output_scaler_path))

            return input_scaler, output_scaler
        else:
            print("Did not create dataset")
            return None, None

    def _make_data(self):
        trials_folders = subdirs(Path(self.data_path))

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
                raise ValueError

            # Get trajectory point at each simulation step
            traj_sim = trajectory_at_each_simulation_step(trajectory, history)
            delta_traj = traj_sim[1:, :] - traj_sim[:-1, :]
            norm_input = input_scaler.transform(delta_traj)

            out = np.vstack([history["tau_r"][1:], history["tau_l"][1:]]).T
            out[out > self.trim_controls] = self.trim_controls
            out[out < -self.trim_controls] = -self.trim_controls
            norm_output = output_scaler.transform(out)

            # Append to dataset
            data["trajectory"].append(norm_input)
            data["tau_r"].append(norm_output[:, 0])
            data["tau_l"].append(norm_output[:, 1])
            data["sim_dt"].append(config["dt"])

        # Save data to file
        self.plot_dataset(
            np.vstack(data["trajectory"]),
            np.vstack(
                [np.concatenate(data["tau_r"]), np.concatenate(data["tau_l"])]
            ).T,
        )

        if Confirm.ask("Save dataset?", default=True):
            pd.DataFrame(data).to_hdf(self.dataset_path, key="hdf")
            self._data = pd.read_hdf(self.dataset_path, key="hdf")
            self._n_trials = len(data["sim_dt"])
            print(f"Saved at {self.dataset_path}, {self._n_trials} trials")
        else:
            print("Did not save dataset")

    def generate_trial_params(self, batch, trial):
        """"Define parameters for each trial.

        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.

        Returns:
            dict: Dictionary of trial parameters.
            
        """

        # Get a random trial dir
        trial_n = np.random.randint(0, self._n_trials)

        return dict(
            trajectory=self._data["trajectory"].values[trial_n],
            tau_r=self._data["tau_r"].values[trial_n],
            tau_l=self._data["tau_l"].values[trial_n],
            sim_dt=self._data["sim_dt"].values[trial_n],
        )

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.

        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.

        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()

        Returns:
            tuple:

            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t,
                            False if the network should ignore y_t when training.
        
        """

        # Get the simulation step
        N = len(params["trajectory"])
        step = np.int(np.floor((N * time) / self.T))

        x_t = params["trajectory"][step, :]
        y_t = np.hstack([params["tau_r"][step], params["tau_l"][step]])
        return x_t, y_t, np.ones(self.N_out)
