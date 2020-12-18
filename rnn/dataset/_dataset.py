import pandas as pd
import numpy.random as rnd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from rich.progress import track
from rich import print
import numpy as np
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from random import choice
from scipy.signal import resample

from pyrnn._utils import torchify
from pyinspect.utils import subdirs
from myterial import (
    orange,
    salmon,
    teal,
    light_blue,
    indigo,
    green_dark,
    blue_grey,
)

from rnn.paths import RNNPaths
from control.history import load_results_from_folder

"""
    Preprocess results from running the control
    algorithm on a number of trials to create a 
    normalized dataset for using with RNNs.
"""

colors = (orange, salmon, teal, light_blue, indigo, green_dark, blue_grey)

# ---------------------------------------------------------------------------- #
#                                    DATASET                                   #
# ---------------------------------------------------------------------------- #


class Dataset(data.Dataset, RNNPaths):
    # variables controlling trials creation
    # may be overwritten by train_params.py
    augment_probability = 0
    to_chunks = False
    chunk_length = 128
    warmup = False

    def __init__(self, dataset_length=-1, **kwargs):
        RNNPaths.__init__(self, dataset_name=self.name, **kwargs)

        try:
            self.dataset = pd.read_hdf(self.dataset_train_path, key="hdf")[
                :dataset_length
            ]

            self.inputs = self.dataset[list(self.inputs_names)]
            self.outputs = self.dataset[list(self.outputs_names)]
        except FileNotFoundError:
            print("No data to load")

    def __len__(self):
        return len(self.dataset)

    def _get_random(self):
        """
            1. get a random trial from dataset
            2. pad it
            3. enjoy
        """
        idx = rnd.randint(0, len(self))
        X, Y = self.__getitem__(idx)

        x_padded = pad_sequence([X], batch_first=True, padding_value=0)
        y_padded = pad_sequence([Y], batch_first=True, padding_value=0)

        return x_padded, y_padded

    def _slice(self, X, Y):
        """ Augment data by taking part of a trial """
        l = len(X)
        start = rnd.randint(1, l - 10)

        X, Y = X[start:, :], Y[start:, :]

        if len(X.shape) == 1 or len(Y.shape) == 1 or X.shape[0] != Y.shape[0]:
            raise ValueError("Error while augmenting data")
        return X, Y

    def _speed(self, X, Y, factor):
        """ Augment data by speedin up/down trials """
        l = int(len(X) * factor)

        X = torchify(resample(X, l))
        Y = torchify(resample(Y, l))

        # Don't return the whole thing to remove artifacts
        if factor > 1:
            return X[200:-200, :], Y[200:-200, :]
        else:
            return X[50:-50, :], Y[50:-50, :]

    def _augment(self, X, Y):
        """
            General method for augmenting data
        """
        method = choice(("slice", "speed up", "slow down"))

        if method == "slice":
            return self._slice(X, Y)
        elif method == "speed up":
            return self._speed(X, Y, 0.5)
        elif method == "slow down":
            return self._speed(X, Y, 2)

    def _add_warmup(self, X, Y):
        """ add a warmup phase of constant inputs 
            at start of trial to facilitate learning """
        l = len(X)
        warmup = 10

        x = np.zeros((warmup + l, X.shape[1]))
        x[:warmup, :] = X[0, :]
        x[warmup:, :] = X

        y = np.zeros((warmup + l, Y.shape[1]))
        y[:warmup, :] = Y[0, :]
        y[warmup:, :] = Y

        return torchify(x), torchify(y)

    def chunk(self, X, Y):
        """
            Take a random chunk of L length from a trial, to train on smaller
            chunks
        """
        l = len(X)
        if l <= self.chunk_length + 2:
            return X, Y
        start = rnd.randint(0, l - self.chunk_length - 1)

        return (
            X[start : start + self.chunk_length, :],
            Y[start : start + self.chunk_length, :],
        )

    def __getitem__(self, item):
        """
            Get a single trial
        """
        X = torchify(np.vstack(self.inputs.iloc[item].values).T)
        Y = torchify(np.vstack(self.outputs.iloc[item].values).T)

        if len(X) != len(Y):
            raise ValueError("Length of X and Y must match")

        # augment data
        if rnd.random() <= self.augment_probability:
            X, Y = self._augment(X, Y)

        # add 'warm up'
        if self.warmup:
            X, Y = self._add_warmup(X, Y)

        # chunk trial
        if self.to_chunks:
            X, Y = self.chunk(X, Y)

        return X, Y

    def plot_random(self):
        """
            Plots a random trial to inspect the content of the dataset
        """
        X, Y = self._get_random()

        f, axarr = plt.subplots(nrows=3, figsize=(14, 8), sharex=False)
        axarr[0].plot(X[0, :, 0], X[0, :, 1])

        for n, name in enumerate(self.inputs_names):
            axarr[1].plot(X[0, :, n], lw=2, label=name, color=colors[n])

        for m, name in enumerate(self.outputs_names):
            axarr[2].plot(Y[0, :, m], lw=2, label=name, color=colors[n + m])

        axarr[1].legend()
        axarr[2].legend()

        plt.show()

    def plot_durations(self):
        f, ax = plt.subplots()
        durs = [len(x.x) for i, x in self.inputs.iterrows()]
        ax.hist(durs)
        ax.set(title="Trials durations")

        plt.show()

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


# ---------------------------------------------------------------------------- #
#                                 PREPROCESSING                                #
# ---------------------------------------------------------------------------- #


class Preprocessing(RNNPaths):
    """
        Class to take the results of iLQR and organize them
        into a structured dataset that can be used for training RNNs
    """

    name = "dataset name"
    description = "base"  # updated in subclasses to describe dataset

    # names of inputs and outputs of dataset
    inputs_names = ("x", "y", "theta", "v", "omega")
    outputs_names = ("out1", "out2")

    def __init__(self, test_size=0.1, truncate_at=None, **kwargs):
        RNNPaths.__init__(self, dataset_name=self.name, **kwargs)
        self.test_size = test_size
        self.truncate_at = truncate_at

    def get_inputs(self, trajectory, history):
        return NotImplementedError(
            "get_inputs should be implemented in your dataset preprocessing"
        )
        # should return x,y,theta,v,omega

    def get_outputs(self, history):
        return NotImplementedError(
            "get_outputs should be implemented in your dataset preprocessing"
        )
        # should return output1, output2

    def truncate(self, train, test):
        """
            Splits each trial into
            chunks, all chunks will
            have the same length, the number of chunks
            per trial depends on the trial length.
        """

        def run_one(df):
            truncated = {k: [] for k in df.columns}
            for i, t in df.iterrows():
                n = t.x.shape[0]
                n_chunks = int(np.floor(n / self.truncate_at)) - 1

                for chunk in np.arange(n_chunks):
                    for k in df.columns:
                        truncated[k].append(
                            t[k][
                                self.truncate_at
                                * chunk : self.truncate_at
                                * (chunk + 1)
                            ]
                        )

            return pd.DataFrame(truncated)

        truncated_train = run_one(train)
        truncated_test = run_one(test)

        return truncated_train, truncated_test

    def fit_scaler(self, df):
        # concatenate the values under each columns to fit a scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data = pd.DataFrame(
            {c: np.concatenate(df[c].values) for c in df.columns}
        )
        return scaler.fit(data)

    def scale(self, df, scaler):
        """
            Use a fitted minmax scaler to scale
            each trial in a dataframe
        """
        scaled = {c: [] for c in df.columns}
        for i, t in df.iterrows():
            scld = scaler.transform(np.vstack(t.values).T)
            for n, c in enumerate(df.columns):
                scaled[c].append(scld[:, n])

        return pd.DataFrame(scaled)

    def split_and_normalize(self, data):
        train, test = train_test_split(data, test_size=self.test_size)

        train_scaler = self.fit_scaler(train)
        train = self.scale(train, train_scaler)

        test_scaler = self.fit_scaler(train)
        test = self.scale(test, test_scaler)

        self.save_normalizers(train_scaler, test_scaler)
        return train, test

    def describe(self):
        with open(self.dataset_folder / "description.txt", "w") as out:
            out.write(self.description)

    def make(self):
        """ 
            Organizes the standardized data into a single dataframe.
        """
        trials_folders = subdirs(self.trials_folder)
        print(
            f"[bold magenta]Creating dataset...\nFound {len(trials_folders)} trials folders."
        )

        # Create dataframe with all trials
        data = {
            **{k: [] for k in self.inputs_names},
            **{k: [] for k in self.outputs_names},
        }
        for fld in track(trials_folders):
            try:
                (history, info, trajectory, trial) = load_results_from_folder(
                    fld
                )
            except Exception as e:
                print(
                    f"Could not open a trial folder, skipping: {fld.name}:\n   {e}"
                )
                continue

            # Get inputs
            inputs = self.get_inputs(trajectory, history)
            for name, value in zip(self.inputs_names, inputs):
                data[name].append(value)

            # get outputs
            outputs = self.get_outputs(history)
            for name, value in zip(self.outputs_names, outputs):
                data[name].append(value)

        # as dataframe
        data = pd.DataFrame(data)

        # split and normalize
        train, test = self.split_and_normalize(data)

        # truncate if necessary
        if self.truncate_at is not None:
            train, test = self.truncate(train, test)

        # save
        self.save_dataset(train, test)
        self.describe()
