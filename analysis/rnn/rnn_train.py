import matplotlib.pyplot as plt
from pyrnn import RNN
from pyrnn.plot import plot_training_loss
import os
import sys

sys.path.append("./")

from proj.rnn.dataset import PredictNudotFromXYT, plot_predictions

MAKE_DATASET = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------- Preprocess dataset ---------------------------- #
if MAKE_DATASET:
    PredictNudotFromXYT(truncate_at=None).make()

# ---------------------------------- Params ---------------------------------- #
name = None
batch_size = 2048
epochs = 500  # 300
lr_milestones = [150, 450]
lr = 0.01
stop_loss = 0.002

# ------------------------------- Fit/load RNN ------------------------------- #
if not MAKE_DATASET:
    rnn = RNN(
        input_size=3,
        output_size=2,
        n_units=64,
        dale_ratio=None,
        autopses=True,
        on_gpu=True,
        # w_in_bias=True,
        # w_in_train=True,
        # w_out_bias=True,
        # w_out_train=True,
    )

    loss_history = rnn.fit(
        PredictNudotFromXYT(),
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        lr_milestones=lr_milestones,
        l2norm=0,
        stop_loss=stop_loss,
    )
    rnn.save("task_rnn.pt")

    plot_predictions(rnn, batch_size, PredictNudotFromXYT)
    plot_training_loss(loss_history)
    plt.show()
