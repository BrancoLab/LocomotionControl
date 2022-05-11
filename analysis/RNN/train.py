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

# save_folder = Path(
#     "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/RNN/trained_networks"
# )
save_folder = Path(
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\RNN\trained_networks"
)

plt.ion()

# ---------------------------------- params ---------------------------------- #
n_units = 128
batch_size = 512
epochs = 5000
lr_milestones = [2500, 5000, 350_000, 450_000]
lr = 0.25

# ---------------------------------- Fit RNN --------------------------------- #

dataset = GoalDirectedLocomotionDataset(max_dataset_length=512)


rnn = RNN(
    input_size=dataset.n_inputs,
    output_size=dataset.n_outputs,
    autopses=True,
    dale_ratio=None,
    n_units=n_units,
    on_gpu=True,
    w_in_train=True,
    w_out_train=True,
    tau=50,
    dt=5,
)

try:
    loss_history = rnn.fit(
        dataset,
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_length=dataset.sequence_length,
        lr_milestones=lr_milestones,
        l2norm=0,
        save_at_min_loss=True,
        save_path=str(save_folder / "gdl_minloss.pt"),
    )

except KeyboardInterrupt:
    rnn.save(str(save_folder / "gld.pt"))
    plot_predictions(rnn)
    plt.show()
