"""
    Training of a simple RNN that receives trajectories 
    info (cartesian model) and learns to predict the output predicted by the 
    optimal control cartesian model

    Install pythorch from: https://pytorch.org/get-started/locally/

    example code: https://github.com/FedeClaudi/wheel_control_rnn/blob/master/learn_rnn_me.py
"""
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import random
import pyinspect as pi
from pathlib import Path
from rich.progress import track

pi.install_traceback()

import sys

sys.path.append("./")
from proj import paths
from proj.utils.misc import load_results_from_folder

# %%
# parameters
BATCH_SIZE = 64  # our batch size is fixed for now
N_INPUT = 80  # length of the input vector
N_VECS = 5  # dimensionality of input
N_NEURONS = 100
N_EPOCHS = 25
DATASET_LENGTH = 10000


# Other params
MAX_TRAJ_LEN = 800  # only simulation in which the model reached the goal within N steps are accepted
# all simulations longer than this are rejected
MIN_TRAJ_LEN = (
    N_INPUT  # 400  # trajectories are truncated to be of this length
)

# %%
# ---------------------------------------------------------------------------- #
#                                   MAKE DATA                                  #
# ---------------------------------------------------------------------------- #


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        vec, label = sample["vec"], sample["label"]

        return {
            "vec": torch.from_numpy(np.array(vec)).to(dtype=torch.float),
            "label": torch.from_numpy(np.array(label)).to(dtype=torch.float),
        }


class MyData(Dataset):
    def __init__(self, length, veclen, n_vecs, transform=False):
        self.length = length
        self.veclen = veclen
        self.transform = transform
        self.n_vecs = n_vecs

        self.simulation_folders = [
            f for f in Path(paths.db_app).glob("*") if f.is_dir()
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load data
        while True:  # keep trying until we get a trajectory with all samples

            # Get a simulation folder and load data
            fld = random.choice(self.simulation_folders)

            try:
                (
                    config,
                    trajectory,
                    history,
                    cost_history,
                ) = load_results_from_folder(fld)
            except ValueError:
                continue

            # Get controls history
            controls = np.vstack(history[["tau_r", "tau_l"]].values)

            # Get 'inputs' that resulted in the controls
            inputs = np.vstack(
                [trajectory[s, :] for s in history.trajectory_idx]
            )

            if (
                controls.shape[0] > MAX_TRAJ_LEN
                or controls.shape[0] < MIN_TRAJ_LEN
            ):
                # too long or too short
                continue
            else:
                # truncate
                controls = controls[:MIN_TRAJ_LEN, :]
                inputs = inputs[:MIN_TRAJ_LEN, :]
                break  # exit loop

        sample = {"vec": inputs, "label": controls}

        if self.transform:
            sample = self.transform(sample)

        return sample


v, i = MyData(DATASET_LENGTH, N_INPUT, N_VECS, transform=ToTensor())[
    0
].values()

pi.ok("Data model created", f"Input {v.shape}\nLabel {i.shape}")


# %%

# ---------------------------------------------------------------------------- #
#                                 Define model                                 #
# ---------------------------------------------------------------------------- #


class Model(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons):
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        self.rnn = nn.RNN(self.n_inputs, self.n_neurons, nonlinearity="relu")

    def init_hidden(self,):
        # (num_layers, batch_size, n_neurons)
        return torch.zeros(1, self.batch_size, self.n_neurons)

    def forward(self, X):
        # Reshape X: n_steps X batch_size X n_inputs
        # X = X.unsqueeze(0)
        # X = X.permute((1, 0, 2))
        X = X.permute((2, 0, 1))

        # for each time step
        self.hidden = self.rnn(X, self.hidden)

        return [
            s.reshape(self.batch_size, self.n_neurons) for s in self.hidden
        ]


# %%
# ---------------------------------------------------------------------------- #
#                                   TRAINING                                   #
# ---------------------------------------------------------------------------- #ˆ
# Get model
model = Model(BATCH_SIZE, N_INPUT, N_NEURONS)

# Get optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get loss function
lossfn = nn.MSELoss()


def getloss(activations, target):
    """
        Output is the sum of each units' activation
        and needs to match the target
    """
    return lossfn(activations.sum(axis=1), target)


# Get dataset
mydata = MyData(DATASET_LENGTH, N_INPUT, N_VECS, transform=ToTensor())
dataloader = DataLoader(
    mydata, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
)


pi.ok("Ready to train")

# %%
# ----------------------------------- Train ---------------------------------- #
loss_record = []
for epoch in track(range(N_EPOCHS), total=N_EPOCHS):
    train_running_loss = 0.0
    model.train()

    # TRAINING ROUND
    for i, data in enumerate(dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # get the inputs
        inputs, labels = data.values()

        # reset hidden states
        model.hidden = model.init_hidden()

        # forward
        output, hidden = model(inputs)  # returns the inner state

        # Compute loss
        loss = getloss(hidden, labels)

        # backward and SGD
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()

    model.eval()
    if epoch % 10 == 0:
        print(f"Epoch:  {epoch} | Loss: {round(train_running_loss, 2)}")
    loss_record.append(train_running_loss)

plt.plot(loss_record)


# %%

# %%
