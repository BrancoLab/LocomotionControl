from psychrnn.tasks.task import Task
from pyinspect.utils import subdirs
import numpy as np
from pathlib import Path

from random import choice, choices

from proj.paths import rnn_trainig
from proj.utils.misc import load_results_from_folder


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
            data_path = rnn_trainig
        self.trials_folders = subdirs(Path(data_path))

        n_test_set = int(len(self.trials_folders) / 3)

        self.train_set = choices(
            self.trials_folders, k=len(self.trials_folders) - n_test_set
        )
        self.test_set = [
            f for f in self.trials_folders if f not in self.train_set
        ]

    def generate_trial_params(self, batch, trial):
        """"Define parameters for each trial.

        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.

        Returns:
            dict: Dictionary of trial parameters.
            
        """

        # Get a random trial dir
        tdir = choice(self.train_set)

        (
            config,
            trajectory,
            history,
            cost_history,
            trial,
            info,
        ) = load_results_from_folder(tdir)

        # Get trajectory point at each simulation step
        traj_sim = np.vstack(
            [trajectory[i, :] for i in history.trajectory_idx]
        )

        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        return dict(
            trajectory=traj_sim,
            tau_r=history["tau_r"],
            tau_l=history["tau_l"],
            sim_dt=config["dt"],
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
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
        
        """

        # Get the simulation step
        N = len(params["trajectory"])
        step = np.int(np.floor((N * time) / self.T))

        if step >= N - 1:
            x_t = params["trajectory"][-1, :] - params["trajectory"][-2, :]
        else:
            x_t = (
                params["trajectory"][step + 1, :]
                - params["trajectory"][step, :]
            )
        y_t = np.hstack([params["tau_r"][step], params["tau_r"][step]])

        return x_t, y_t, True
