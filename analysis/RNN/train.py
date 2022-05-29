import matplotlib.pyplot as plt
from torch import nn
import sys

sys.path.append("./")
from pathlib import Path

from pyrnn import RNN
from analysis.RNN.task import GoalDirectedLocomotionDataset

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# save_folder = Path(
#     "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/RNN/trained_networks"
# )
save_folder = Path(
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\RNN\trained_networks"
)

plt.ion()


# ---------------------------------- params ---------------------------------- #
n_units = 64
batch_size = 256
epochs = 30_000
lr_milestones = None  # [2000, 50000, 10000000]
lr = 0.005
save_every = 5000

PLANNING_HORIZON = 0  # cm

# ---------------------------------- Fit RNN --------------------------------- #

dataset = GoalDirectedLocomotionDataset(
    max_dataset_length=3000, horizon=PLANNING_HORIZON
)


rnn = RNN(
    input_size=dataset.n_inputs,
    output_size=dataset.n_outputs,
    autopses=True,
    dale_ratio=None,
    n_units=n_units,
    on_gpu=True,
    w_in_train=True,
    w_out_train=True,
)


# train
try:
    loss_history = rnn.fit(
        dataset,
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_length=dataset.sequence_length,
        lr_milestones=lr_milestones,
        l2norm=5e-5,
        save_at_min_loss=True,
        save_path=save_folder,
        loss_fn=nn.SmoothL1Loss,
        save_every=save_every,
        gamma=0.5,
        save_name=f"rnn_{PLANNING_HORIZON}cm",
    )

except KeyboardInterrupt:
    pass


# save data normalizers and dataset metatapython add_data
dataset.save_normalizers(rnn.recorder.folder)
rnn.recorder.add_data(dataset.metadata, "dataset_metadata", fmt="yaml")
