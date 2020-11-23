import matplotlib.pyplot as plt

from pyrnn import RNN
from pyrnn.plot import plot_training_loss

import sys

sys.path.append("./")
from proj.rnn.dataset import (
    TrajAtEachFrame,
    plot_predictions,
)


# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------- Preprocess dataset ---------------------------- #
# from proj.rnn.preprocess_dataset import PredictNudotFromDeltaXYT
# PredictNudotFromDeltaXYT().make()

# ---------------------------------- Params ---------------------------------- #
name = None
batch_size = 512
epochs = 2000  # 300
lr_milestones = [10, 1000, 1900]
lr = 0.01
stop_loss = 0.002

# ------------------------------- Fit/load RNN ------------------------------- #

rnn = RNN(
    input_size=3,
    output_size=2,
    n_units=64,
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
