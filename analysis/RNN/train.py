import matplotlib.pyplot as plt
import os

import sys

sys.path.append("./")
from pathlib import Path

from pyrnn import CTRNN as RNN
from analysis.RNN.task import (
    GoalDirectedLocomotionDataset,
    plot_predictions,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

save_folder = Path(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/RNN/trained_networks"
)


# ---------------------------------- params ---------------------------------- #
n_units = 64
batch_size = 256
epochs = 2000
lr_milestones = None  # [1000, 2000, 5000, 6000]
lr = 0.0025

# ---------------------------------- Fit RNN --------------------------------- #

dataset = GoalDirectedLocomotionDataset()


rnn = RNN(
    input_size=dataset.n_inputs,
    output_size=dataset.n_outputs,
    autopses=True,
    dale_ratio=None,
    n_units=n_units,
    on_gpu=False,
    w_in_train=True,
    w_out_train=True,
    tau=50,
    dt=5,
)

loss_history = rnn.fit(
    dataset,
    n_epochs=epochs,
    lr=lr,
    batch_size=batch_size,
    input_length=dataset.sequence_length,
    lr_milestones=lr_milestones,
    l2norm=0,
    save_at_min_loss=True,
    save_path=save_folder / "gdl_minloss.pt",
)
rnn.save(save_folder / "gld.pt")

plot_predictions(rnn)
plt.show()
