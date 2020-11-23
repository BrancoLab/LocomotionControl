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
from torch.nn.utils.rnn import pad_sequence
import sys
import pandas as pd

from proj.rnn._utils import RNNPaths

from pyrnn._plot import clean_axes
from pyrnn._utils import torchify


is_win = sys.platform == "win32"


class DataSet(data.Dataset, RNNPaths):
    # preprocessed dataset name
    dataset_name = "replace"

    # input and outputs names
    _data = (("x", "y", "theta"), ("nudot_R", "nudot_L"))

    def __init__(self, dataset_length=-1):
        RNNPaths.__init__(self, dataset_name=self.dataset_name)

        self.dataset = pd.read_hdf(self.dataset_train_path, key="hdf")[
            :dataset_length
        ]
        self.get_max_trial_length()

        self.inputs = self.dataset[list(self._data[0])]
        self.outputs = self.dataset[list(self._data[1])]

    def __len__(self):
        return len(self.dataset)

    def get_max_trial_length(self):
        self.n_samples = max(
            [len(t[self._data[0][0]]) for i, t in self.dataset.iterrows()]
        )

    def _pad(self, arr):
        arr = np.vstack(arr).T
        l, m = arr.shape
        padded = np.zeros((self.n_samples, m))
        padded[:l, :] = arr
        return padded

    def _get_random(self):
        idx = rnd.randint(0, len(self))
        X, Y = self.__getitem__(idx)

        X = torchify(self._pad(X)).reshape(1, self.n_samples, -1)
        Y = torchify(self._pad(Y)).reshape(1, self.n_samples, -1)

        return X, Y

    def __getitem__(self, item):
        """
            1. get a random trial from dataset
            2. shape and pad it
            3. create batch
            4. enjoy
        """
        X = torchify(np.vstack(self.inputs.iloc[item].values).T)
        Y = torchify(np.vstack(self.outputs.iloc[item].values).T)

        if len(X) != len(Y):
            raise ValueError("Length of X and Y must match")

        return X, Y

    @classmethod
    def get_one_batch(cls, n_trials, **kwargs):
        """
        Return a single batch of given length    
        """
        ds = cls(dataset_length=n_trials, **kwargs)
        batch = [b for b in ds]

        x_padded = pad_sequence(
            [b[0] for b in batch], batch_first=True, padding_value=0
        )
        y_padded = pad_sequence(
            [b[1] for b in batch], batch_first=True, padding_value=0
        )

        return x_padded, y_padded


class TrajAtEachFrame(DataSet):
    """
    creates a pytorch dataset for loading
    the data during training.
    """

    # preprocessed dataset name
    dataset_name = "dataset_predict_nudot_from_deltaXYT"

    # input and outputs names
    _data = (("x", "y", "theta"), ("nudot_R", "nudot_L"))

    def __init__(self, *args, **kwargs):
        DataSet.__init__(self, *args, **kwargs)


def plot_predictions(model, batch_size, dataset, **kwargs):
    """
    Run the model on a single batch and plot
    the model's prediction's against the
    input data and labels.
    """
    X, Y = dataset.get_one_batch(1, **kwargs)

    if model.on_gpu:
        model.cpu()
        model.on_gpu = False

    o, h = model.predict(X)

    n_inputs = X.shape[-1]
    n_outputs = Y.shape[-1]
    labels = ["x", "y", "$\\theta$", "v", "$\\omega$"]

    f, axarr = plt.subplots(nrows=2, figsize=(12, 9))

    for n in range(n_inputs):
        axarr[0].plot(X[0, :, n], lw=2, label=labels[n])
    axarr[0].set(title="inputs")
    axarr[0].legend()

    cc = [salmon, light_green]
    oc = [salmon_dark, light_green_dark]
    labels = ["nudot_R", "nudot_L"]
    for n in range(n_outputs):
        axarr[1].plot(
            Y[0, :, n], lw=2, color=cc[n], label="correct " + labels[n]
        )
        axarr[1].plot(
            o[0, :, n], lw=2, ls="--", color=oc[n], label="model output"
        )
    axarr[1].legend()
    axarr[1].set(title="outputs")

    f.tight_layout()
    clean_axes(f)
