import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
from myterial import (
    salmon,
    salmon_dark,
    light_green,
    light_green_dark,
)
import torch.utils.data as data
import sys
import pandas as pd

from proj.rnn._utils import RNNPaths

from pyrnn._plot import clean_axes
from pyrnn._utils import torchify


is_win = sys.platform == "win32"


class DeltaStateDataSet(data.Dataset, RNNPaths):
    """
    creates a pytorch dataset for loading
    the data during training.
    """

    def __init__(self, name=None, dataset_length=-1):
        self.name = name or "test"
        RNNPaths.__init__(self)

        self.dataset = pd.read_hdf(self.dataset_train_path, key="hdf")[
            :dataset_length
        ]
        self.get_max_trial_length()

    def __len__(self):
        return len(self.dataset)

    def get_max_trial_length(self):
        self.n_samples = max(
            [len(t.trajectory) for i, t in self.dataset.iterrows()]
        )

    def _pad(self, arr):
        l, m = arr.shape
        padded = np.zeros((self.n_samples, m))
        padded[:l, :] = arr
        return padded

    def _get_random(self):
        idx = rnd.randint(0, len(self))
        trial = self.dataset.iloc[idx]

        X = torchify(self._pad(trial.trajectory))
        Y = torchify(self._pad(trial.controls))

        if len(X) != len(Y):
            raise ValueError("Length of X and Y must match")

        return (
            X.reshape(1, self.n_samples, -1),
            Y.reshape(1, self.n_samples, -1),
        )

    def __getitem__(self, item):
        """
            1. get a random trial from dataset
            2. shape and pad it
            3. create batch
            4. enjoy
        """
        trial = self.dataset.iloc[item]

        X = torchify(self._pad(trial.trajectory))
        Y = torchify(self._pad(trial.controls))

        if len(X) != len(Y):
            raise ValueError("Length of X and Y must match")

        return X, Y


def make_batch(**kwargs):
    """
    Return a single batch of given length    
    """
    batch = DeltaStateDataSet(dataset_length=500, **kwargs)._get_random()
    return batch


def plot_predictions(model, batch_size, **kwargs):
    """
    Run the model on a single batch and plot
    the model's prediction's against the
    input data and labels.
    """
    X, Y = make_batch(**kwargs)
    o, h = model.predict(X)

    n_inputs = X.shape[-1]
    n_outputs = Y.shape[-1]

    f, axarr = plt.subplots(nrows=2, figsize=(12, 9))

    for n in range(n_inputs):
        axarr[0].plot(X[0, :, n], lw=2)
    axarr[0].set(title="inputs")

    cc = [salmon, light_green]
    oc = [salmon_dark, light_green_dark]
    for n in range(n_outputs):
        axarr[1].plot(Y[0, :, n], lw=2, color=cc[n], label="correct")
        axarr[1].plot(
            o[0, :, n], lw=1, ls="--", color=oc[n], label="model output"
        )
    axarr[1].legend()
    axarr[1].set(title="outputs")

    f.tight_layout()
    clean_axes(f)
