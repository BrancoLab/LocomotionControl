from psychrnn.tasks.task import Task

from pyinspect.utils import subdirs
import numpy as np
from pathlib import Path
from rich import print
from rich.progress import track
import pandas as pd

from proj.paths import rnn_trainig
from proj.utils.misc import load_results_from_folder
from sklearn.preprocessing import MinMaxScaler


class ControlTask(Task):
    def __init__(
        self, dt, tau, T, N_batch, n_inputs=5, n_outputs=2, data_path=None
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

        if data_path is None:
            self.data_path = rnn_trainig
        else:
            self.data_path = data_path

        self.data_store = Path(self.data_path).parent / "training_data.h5"

        try:
            self._data = pd.read_hdf(self.data_store, key="hdf")
            self._n_trials = len(self._data)
        except FileNotFoundError:
            print("Did not find data file, make data?")
            self._make_data()

    def _make_data(self):
        print("Making")
        self.trials_folders = subdirs(Path(self.data_path))

        print(f"Len dataset = {len(self.trials_folders)}")
        params = dict(trajectory=[], tau_r=[], tau_l=[], sim_dt=[],)

        print("Creating training data")
        for fld in track(self.trials_folders):
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
            traj_sim = np.vstack(
                [
                    trajectory[i, :]
                    for i in history.trajectory_idx
                    if i < len(trajectory) - 50
                ]  # ! skipping the end artefacts
            )

            # normalize inputs and outputs
            input_scaler = MinMaxScaler(feature_range=(0, 1))
            input_scaler = input_scaler.fit(traj_sim)
            normalized = input_scaler.transform(traj_sim)

            output_scaler = MinMaxScaler(feature_range=(0, 1))
            output = np.vstack(
                [
                    history["tau_r"][: len(traj_sim)],
                    history["tau_l"][: len(traj_sim)],
                ]
            )
            output_scaler = output_scaler.fit(output)
            norm_output = output_scaler.transform(output)

            params["trajectory"].append(normalized)
            params["tau_r"].append(norm_output[:, 0])
            params["tau_l"].append(norm_output[:, 1])
            params["sim_dt"].append(config["dt"])

        pd.DataFrame(params).to_hdf(self.data_store, key="hdf")
        self._data = pd.read_hdf(self.data_store, key="hdf")
        self._n_trials = len(self.trials_folders)

    def generate_trial_params(self, batch, trial):
        """"Define parameters for each trial.

        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.

        Returns:
            dict: Dictionary of trial parameters.
            
        """

        # Get a random trial dir
        # trial_n = np.random.randint(0, self._n_trials)
        # trial_n = 1
        trial_n = (self.N_batch * batch) + trial

        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
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

        # if step >= N - 2:
        #     x_t = params["trajectory"][-1, :] - params["trajectory"][-2, :]
        # else:
        #     x_t = (
        #         params["trajectory"][step + 1, :]
        #         - params["trajectory"][step, :]
        #     )
        x_t = params["trajectory"][step, :]
        y_t = np.hstack([params["tau_r"][step], params["tau_r"][step]])

        return x_t, y_t, np.ones(self.N_out)
