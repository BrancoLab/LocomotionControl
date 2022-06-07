import matplotlib.pyplot as plt
from torch import nn
import sys
from torch.utils.data import Subset
from fcutils.path import subdirs
from sklearn.model_selection import train_test_split
from loguru import logger


sys.path.append("./")
from pathlib import Path

from pyrnn import RNN
from analysis.RNN.task import GoalDirectedLocomotionDataset


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split
    )
    datasets = {}
    datasets["train"] = Subset(dataset, train_idx)
    datasets["test"] = Subset(dataset, val_idx)
    return datasets


# ---------------------------------- params ---------------------------------- #
n_units = 256
batch_size = 4048
epochs = 1000
lr_milestones = None  # [2000, 50000, 10000000]
lr = 4e-4
save_every = 5000


# ----------------------------------- paths ---------------------------------- #
save_folder = Path(
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\RNN\datasets_comparison"
)
datasetes_folder = Path(
    r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\RNN\datasets"
)


# ---------------------------------------------------------------------------- #
#                                   RUN SIMS                                   #
# ---------------------------------------------------------------------------- #
plt.ion()
for dataset_dir in subdirs(datasetes_folder):

    if (save_folder / dataset_dir.name).exists():
        if (save_folder / dataset_dir.name / "training_loss.npy").exists():
            logger.warning(f"Skippin previously run folder: {dataset_dir}")
            continue

    logger.info(f"Doing {dataset_dir}")

    dataset = GoalDirectedLocomotionDataset(data_folder=dataset_dir)

    # split train/test sets
    datasets = train_val_dataset(dataset)

    # ---------------------------------- Fit RNN --------------------------------- #
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

    rnn.fit(
        datasets["train"],
        test_dataset=datasets["test"],
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_length=dataset.sequence_length,
        lr_milestones=lr_milestones,
        l2norm=1e-5,
        save_at_min_loss=True,
        save_path=save_folder,
        loss_fn=nn.MSELoss,
        save_every=save_every,
        gamma=0.5,
        save_name=dataset_dir.name,
    )

    # save data normalizers and dataset metatapython add_data
    dataset.save_normalizers(rnn.recorder.folder)
    rnn.recorder.add_data(dataset.metadata, "dataset_metadata", fmt="yaml")
