import pandas as pd
import numpy.random as rnd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from rich.progress import track
from rich import print
import numpy as np
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

from pyrnn._utils import torchify


from pyinspect.utils import subdirs

from proj.rnn._utils import RNNPaths
from proj.utils.misc import load_results_from_folder

"""
    Preprocess results from running the control
    algorithm on a number of trials to create a 
    normalized dataset for using with RNNs.
"""

# ---------------------------------------------------------------------------- #
#                                    DATASET                                   #
# ---------------------------------------------------------------------------- #


class Dataset(data.Dataset, RNNPaths):
    def __init__(self, dataset_length=-1):
        RNNPaths.__init__(self, dataset_name=self.name)

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
    _data = (("x", "y", "theta", "v", "omega"), ("out1", "out2"))

    def __init__(self, test_size=0.1):
        RNNPaths.__init__(self, dataset_name=self.name)
        self.test_size = test_size

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
            **{k: [] for k in self._data[0]},
            **{k: [] for k in self._data[1]},
        }
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
                print(f"Could not open a trial folder, skipping: {fld.name}")
                continue

            # Get inputs
            inputs = self.get_inputs(trajectory, history)
            for name, value in zip(self._data[0], inputs):
                data[name].append(value)

            # get outputs
            outputs = self.get_outputs(history)
            for name, value in zip(self._data[1], outputs):
                data[name].append(value)

        # as dataframe
        data = pd.DataFrame(data)

        # split and normalize
        train, test = self.split_and_normalize(data)

        # save
        self.save_dataset(train, test)
        self.describe()
