import numpy as np
import pandas as pd

from proj.rnn._task import Task
from proj.rnn._utils import RNNLog


class ControlTask(Task, RNNLog):
    def __init__(
        self,
        dt,
        tau,
        T,
        N_batch,
        *args,
        n_inputs=5,
        n_outputs=2,
        test_data=False,
        **kwargs,
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
        Task.__init__(self, n_inputs, n_outputs, dt, tau, T, N_batch)

        mkdir = kwargs.pop("mk_dir", False)
        RNNLog.__init__(self, *args, mk_dir=mkdir, **kwargs)

        try:
            if not test_data:
                self._data = pd.read_hdf(self.dataset_train_path, key="hdf")
            else:
                self._data = pd.read_hdf(self.dataset_test_path, key="hdf")

            self._n_trials = len(self._data)
        except FileNotFoundError:
            raise ValueError("Did not find data file, make data?")

    def generate_trial_params(self, batch, trial):
        """"Define parameters for each trial.

        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.

        Returns:
            dict: Dictionary of trial parameters.
            
        """

        # Get a random trial dir
        if not self.config["single_trial_mode"]:
            trial_n = np.random.randint(0, self._n_trials)
        else:
            trial_n = 0

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

        try:
            x_t = params["trajectory"][step, :]
            y_t = np.hstack([params["tau_r"][step], params["tau_l"][step]])
        except IndexError:
            return 0, 0, np.zeros(self.N_out)
        else:
            return x_t, y_t, np.ones(self.N_out)
