import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from myterial import light_green_dark, indigo_light
from pyrnn._plot import clean_axes
import torch.utils.data as data
import sys
from rich.progress import track
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from random import choice
import joblib


"""
    Locomotion task.

    The RNN receives two sets of inputs at each frame:
        - state in curvilinear coordinates n, psi, s, V, omega
        - track curvature at N points k_1, k_2, ..., k_N equally spaced along the track in fron of the mouse.

    The task is to predict the acceleration and angular acceleration of the mouse.
"""

is_win = sys.platform == "win32"


class GoalDirectedLocomotionDataset(data.Dataset):
    """
    creates a pytorch dataset for loading
    the data during training.
    """

    def __init__(
        self,
        max_dataset_length=-1,
        horizon: int = 50,
        stored_scalers_path=None,
    ):
        self.max_dataset_length = max_dataset_length

        # get paths
        if is_win:
            self.data_folder = Path(
                r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\RNN\datasets"
            )
        else:
            self.data_folder = Path(
                "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/RNN/dataset"
            )
        self.data_folder = self.data_folder / f"{horizon}cm"
        self.horizon = horizon

        # load data
        self.load_data()

        # either load or fit data pre-processing tools
        if stored_scalers_path is None:
            self.fit_normalizers()
        else:
            self.load_normalizers(stored_scalers_path)

        # get n inputs/outputs and sequence length
        self.n_outputs = 2
        self.sequence_length = len(self._raw_data[0]["n"])

        self._inputs = [
            k for k in self._raw_data[0].keys() if k not in ("V̇", "ω̇", "V")
        ]
        self.n_inputs = len(self._inputs)
        self._outputs = ("V̇", "ω̇")
        logger.info(self._inputs)
        logger.info(self._outputs)

        self.make_trials()

    def __len__(self):
        return len(self._raw_train)

    def load_data(self):
        """
        Load the data into the dataset
        """
        logger.info(f"Loading dataset data at: '{self.data_folder}')")

        self._raw_data = [
            pd.read_json(f)
            for f in list(self.data_folder.glob("*.json"))[
                : self.max_dataset_length
            ]
        ]
        if not self._raw_data:
            raise ValueError(f"Did not load any data from {self.data_folder}")

        lengths = [len(t["n"]) for t in self._raw_data]
        assert (
            len(set(lengths)) == 1
        ), f"found trials with unequal length {set(lengths)}"

        # split test/train sets
        I = int(len(self._raw_data) * 0.77)
        self._raw_train, self._raw_test = (
            self._raw_data[:I],
            self._raw_data[I:],
        )

    def fit_normalizers(self):
        """
        Fit the normalization parameters.
        """
        logger.info("Fitting normalization parameters")

        self.normalizers = {
            k: MinMaxScaler(feature_range=[-1, 1])
            for k in self._raw_data[0].columns
        }

        for k, scaler in self.normalizers.items():
            values = np.hstack([t[k].values for t in self._raw_data])
            scaler.fit(values.reshape(-1, 1))

        logger.info(f"Fitted normalizers for: {self.normalizers.keys()}")

    def save_normalizers(self, folder):
        """
        Save the normalizers to disk.
        """
        logger.info("Saving normalizers")
        for k, scaler in self.normalizers.items():
            joblib.dump(scaler, folder / f"{k}_scaler.joblib")

    def load_normalizers(self, folder):
        """
        Load the normalizers from disk.
        """
        logger.info("Loading normalizers")
        self.normalizers = {
            k: joblib.load(folder / f"{k}_scaler.joblib")
            for k in self._raw_data[0].columns
        }

    @property
    def metadata(self):
        """
            A dict with info about the dataset
        """
        return dict(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            sequence_length=self.sequence_length,
            horizon=self.horizon,
            data_flder=self.data_folder,
            inputs=self._inputs,
            outputs=self._outputs,
            n_trials=self.max_dataset_length,
            n_batches=len(self.items),
        )

    def make_trials(self, dataset="train"):
        """
            Generate batches of data.
        """
        logger.info("Generating dataset batches")
        seq_len = self.sequence_length

        if dataset == "train":
            data = self._raw_train
        else:
            data = self._raw_test

        items = {}
        for bn in track(
            range(len(data)),
            description="Generating data...",
            total=len(self),
            transient=True,
        ):
            X_batch = torch.zeros((seq_len, self.n_inputs))
            Y_batch = torch.zeros((seq_len, self.n_outputs))
            trial = data[bn]

            # get inputs
            for i in range(self.n_inputs):
                k = self._inputs[i]
                X_batch[:, i] = torch.tensor(
                    self.normalizers[k]
                    .transform(trial[k].values.reshape(-1, 1))
                    .squeeze()
                )

            # get outputs
            for o in range(self.n_outputs):
                k = self._outputs[o]
                Y_batch[:, o] = torch.tensor(
                    self.normalizers[k]
                    .transform(trial[k].values.reshape(-1, 1))
                    .squeeze()
                )

            items[bn] = (X_batch, Y_batch)

        if dataset == "train":
            self.items = items
        else:
            self.test_items = items

    def __getitem__(self, item):
        X_batch, Y_batch = self.items[item]

        return X_batch, Y_batch


def make_batches(horizon=50):
    """
    Return a single batch of given length
    """
    dataloader = torch.utils.data.DataLoader(
        GoalDirectedLocomotionDataset(max_dataset_length=100, horizon=horizon),
        batch_size=1,
        num_workers=0 if is_win else 2,
        shuffle=True,
        worker_init_fn=lambda x: np.random.seed(),
    )

    batches = [b for b in dataloader]
    return batches


def plot_predictions(model, horizon=50):
    """
    Run the model on a single batch and plot
    the model's prediction's against the
    input data and labels.
    """
    batches = make_batches(horizon=horizon)
    for i in range(3):
        X, Y = choice(batches)
        o, h = model.predict(X)

        f, axarr = plt.subplots(nrows=Y.shape[-1], figsize=(12, 9))

        for n, ax in enumerate(axarr):
            ax.plot(
                Y[0, :, n],
                lw=3,
                color=indigo_light,
                ls="--",
                label="correct output",
            )
            ax.plot(
                o[0, :, n],
                lw=2,
                alpha=0.5,
                color=light_green_dark,
                label="model output",
            )
            ax.set(title=f"Input {n}")
            ax.legend()

        f.tight_layout()
        clean_axes(f)


if __name__ == "__main__":
    make_batches()
