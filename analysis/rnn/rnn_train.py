import matplotlib.pyplot as plt
import os

from pyrnn import RNN
from pyrnn.plot import plot_training_loss

import sys

sys.path.append("./")
# from proj.rnn.preprocess_dataset import PredictNudotFromXYT
from proj.rnn.dataset import (
    TrajAtEachFrame,
    plot_predictions,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------- Preprocess dataset ---------------------------- #
# PredictNudotFromXYT().make()

# ---------------------------------- Params ---------------------------------- #
name = None
batch_size = 512
epochs = 300  # 300
lr_milestones = [100, 200, 1300]
lr = 0.01
stop_loss = 0.002

# ------------------------------- Fit/load RNN ------------------------------- #

rnn = RNN(
    input_size=3,
    output_size=2,
    n_units=128,
    dale_ratio=None,
    autopses=True,
    on_gpu=True,
)

loss_history = rnn.fit(
    TrajAtEachFrame(dataset_length=-1),
    n_epochs=epochs,
    lr=lr,
    batch_size=batch_size,
    lr_milestones=lr_milestones,
    l2norm=0,
    stop_loss=stop_loss,
)
rnn.save("task_rnn.pt")

plot_predictions(rnn, batch_size, TrajAtEachFrame)
plot_training_loss(loss_history)
plt.show()
