import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from typing import List
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.utils.data as data

from fcutils.progress import track
from fcutils.path import from_yaml

from RNN.prepare_data import load_bouts, get_IO_from_bout



"""
    Locomotion task.
        The RNN receives 3 inputs:
            - distance X  milliseconds in the future
            - angle  X  milliseconds in the future
            - orientation X  milliseconds in the future
            - speed current
            - avel current

        and has to produce two outputs:
            - speed next
            - avel next

        In the experiment design both inputs go to MOs while the outputs
        come out of CUN and GRN respectively.

        The outputs have to match those of actual mice.
"""

is_win = sys.platform == "win32"


class LocomotionDataset(data.Dataset):
    """
    creates a pytorch dataset for loading
    the data during training.
    """

    def __init__(self):
        self.params = from_yaml('./RNN/params.yaml')

        self.bouts = load_bouts(keep=5)  # ! for speed, remove whe ready
        self.dataset_length = len(self.bouts)
        self.sequence_length = max([len(b) for b in self.bouts])
        self.make_trials()

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        X_batch, Y_batch = self.items[item]
        return X_batch, Y_batch


    def make_trials(self):
        """
        Generate the set of trials to be used fpr traomomg
        """
        seq_len = self.sequence_length
        dset_len = self.dataset_length

        self.items = {}
        X, Y = [], []
        for i in track(
            range(dset_len),
            description="Fetching data...",
            total=dset_len,
            transient=True,
        ):
            # get bout
            bout = self.bouts[i]
            IO = get_IO_from_bout(bout)

            # initialize empty arrays
            X_batch = np.zeros((len(bout), 5))  # distance, angle in polar coordinates
            Y_batch = np.zeros((len(bout), 2))  # speed, avel

            X_batch[:, 0] = IO['rho']
            X_batch[:, 1] = IO['phi']
            X_batch[:, 2] = IO['theta']
            X_batch[:, 3] = IO['speed']
            X_batch[:, 4] = IO['avel']
            
            Y_batch[:, 0] = IO['target_speed']
            Y_batch[:, 1] = IO['target_avel']

            X.append(X_batch)
            Y.append(Y_batch)

        # normalize data
        X_norm, Y_norm = self.normalize(X, Y)


        # get items 
        for i, (X_batch, Y_batch) in enumerate(zip(X_norm, Y_norm)):
            # # RNN input: batch size * seq len * n_input
            # X = X.reshape(1, seq_len, 1)

            # # out shape = (batch, seq_len, num_directions * hidden_size)
            # Y = Y.reshape(1, seq_len, 1)

            # X_batch[:, m] = X.squeeze()
            # Y_batch[:, m] = Y.squeeze()
            self.items[i] = (X_batch, Y_batch)

    def normalize(self, X:List[np.ndarray], Y:List[np.ndarray]) -> List[np.ndarray]:
        # stack each variable
        RHO = np.hstack([x[:, 0] for x in X])


        # fit minmax scaler
        rho_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(RHO)

        # transform
        X_norm, Y_norm = [], []
        for X_batch, Y_batch in zip(X, Y):
            X_batch_norm = np.zeros_like(X_batch)
            Y_batch_norm = np.zeros_like(Y_batch)

            X_batch_norm[:, 0] = rho_scaler.transform(X_batch[:, 0])

            X_norm.append(X_batch_norm)
            Y_norm.append(Y_batch_norm)

        return X_norm, Y_norm


# def make_batch(seq_len):
#     """
#     Return a single batch of given length
#     """
#     dataloader = torch.utils.data.DataLoader(
#         ThreeBitDataset(seq_len, dataset_length=1),
#         batch_size=1,
#         num_workers=0 if is_win else 2,
#         shuffle=True,
#         worker_init_fn=lambda x: np.random.seed(),
#     )

#     batch = [b for b in dataloader][0]
#     return batch


# def plot_predictions(model, seq_len, batch_size):
#     """
#     Run the model on a single batch and plot
#     the model's prediction's against the
#     input data and labels.
#     """
#     X, Y = make_batch(seq_len)
#     o, h = model.predict(X)

#     f, axarr = plt.subplots(nrows=3, figsize=(12, 9))
#     for n, ax in enumerate(axarr):
#         ax.plot(X[0, :, n], lw=2, color=salmon, label="input")
#         ax.plot(
#             Y[0, :, n],
#             lw=3,
#             color=indigo_light,
#             ls="--",
#             label="correct output",
#         )
#         ax.plot(o[0, :, n], lw=2, color=light_green_dark, label="model output")
#         ax.set(title=f"Input {n}")
#         ax.legend()

#     f.tight_layout()
#     clean_axes(f)
