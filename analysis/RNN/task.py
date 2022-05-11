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

    def __init__(self, max_dataset_length=-1):
        self.max_dataset_length = max_dataset_length

        if is_win:
            self.data_folder = Path(
                r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\RNN\dataset"
            )
        else:
            self.data_folder = Path(
                "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/RNN/dataset"
            )

        self.load_data()

        # get n inputs/outputs and sequence length
        self.n_inputs = len(self._raw_data[0].keys()) - 2
        self.n_outputs = 2
        self.sequence_length = len(self._raw_data[0]["n"])

        self._inputs = [
            k for k in self._raw_data[0].keys() if k not in ("v̇", "ω̇")
        ]
        self._outputs = ("v̇", "ω̇")

        self.make_trials()

    def __len__(self):
        return len(self._raw_data)

    def load_data(self):
        """
        Load the data into the dataset
        """
        logger.info("Loading dataset data")

        self._raw_data = [
            pd.read_json(f)
            for f in list(self.data_folder.glob("*.json"))[
                : self.max_dataset_length
            ]
        ]

        lengths = [len(t["n"]) for t in self._raw_data]
        assert len(set(lengths)) == 1, "found trials with unequal length"

    def make_trials(self):
        """
            Generate batches of data.
        """
        logger.info("Generating dataset batches")
        seq_len = self.sequence_length

        self.items = {}
        for bn in track(
            range(len(self)),
            description="Generating data...",
            total=len(self),
            transient=True,
        ):
            X_batch = torch.zeros((seq_len, self.n_inputs))
            Y_batch = torch.zeros((seq_len, self.n_outputs))
            trial = self._raw_data[bn]

            # get inputs
            for i in range(self.n_inputs):
                k = self._inputs[i]
                X_batch[:, i] = torch.tensor(trial[k])

            # get outputs
            for o in range(self.n_outputs):
                k = self._outputs[o]
                Y_batch[:, o] = torch.tensor(trial[k])

            self.items[bn] = (X_batch, Y_batch)

    def __getitem__(self, item):
        X_batch, Y_batch = self.items[item]

        return X_batch, Y_batch


def make_batch():
    """
    Return a single batch of given length
    """
    dataloader = torch.utils.data.DataLoader(
        GoalDirectedLocomotionDataset(max_dataset_length=1),
        batch_size=1,
        num_workers=0 if is_win else 2,
        shuffle=True,
        worker_init_fn=lambda x: np.random.seed(),
    )

    batch = [b for b in dataloader][0]
    return batch


def plot_predictions(model):
    """
    Run the model on a single batch and plot
    the model's prediction's against the
    input data and labels.
    """
    X, Y = make_batch()
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
        ax.plot(o[0, :, n], lw=2, color=light_green_dark, label="model output")
        ax.set(title=f"Input {n}")
        ax.legend()

    f.tight_layout()
    clean_axes(f)
    plt.show()


if __name__ == "__main__":
    make_batch()
