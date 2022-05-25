import matplotlib.pyplot as plt
from torch import nn
import sys

sys.path.append("./")
from pathlib import Path

from pyrnn import RNN
from analysis.RNN.task import (
    GoalDirectedLocomotionDataset,
    plot_predictions,
)

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# save_folder = Path(
#     "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/RNN/trained_networks"
# )
save_folder = Path(
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\RNN\trained_networks"
)

plt.ion()

# ---------------------------------- params ---------------------------------- #
n_units = 256
batch_size = 256
epochs = 100_000
lr_milestones = [800, 10000, 10000000]
lr = 0.003
save_every = 5000

# ---------------------------------- Fit RNN --------------------------------- #

dataset = GoalDirectedLocomotionDataset(max_dataset_length=3000)


rnn = RNN(
    input_size=dataset.n_inputs,
    output_size=dataset.n_outputs,
    autopses=True,
    dale_ratio=None,
    n_units=n_units,
    on_gpu=True,
    w_in_train=True,
    w_out_train=True,
    # tau=50,
    # dt=5,
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
        save_path=save_folder,
        loss_fn=nn.SmoothL1Loss,
        save_every=save_every,
        gamma=0.5,
    )

except KeyboardInterrupt:
    pass

fig = plot_predictions(rnn)
rnn.recorder.add_figure(fig, "predictions")
