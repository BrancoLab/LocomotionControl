import matplotlib.pyplot as plt
import os
from pyrnn import RNN
from pyrnn.plot import plot_training_loss
from rich import print
from myterial import orange

from rnn.dataset.dataset import PredictNuDotFromXYT as DATASET
from rnn.dataset.dataset import is_win
from rnn.dataset import plot_predictions

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

MAKE_DATASET = False

# ---------------------------- Preprocess dataset ---------------------------- #
if MAKE_DATASET:
    DATASET(truncate_at=None).make()
    DATASET(truncate_at=None).plot_random()
    # DATASET().plot_durations()

# ---------------------------------- Params ---------------------------------- #
n_units = 512

name = DATASET.name
batch_size = 128
epochs = 500  # 300
lr_milestones = [500]
lr = 0.001
stop_loss = 0.002

# ------------------------------- Fit/load RNN ------------------------------- #
if not MAKE_DATASET:
    # Create RNN
    rnn = RNN(
        input_size=len(DATASET.inputs_names),
        output_size=len(DATASET.outputs_names),
        n_units=n_units,
        dale_ratio=None,
        autopses=True,
        w_in_bias=False,
        w_in_train=False,
        w_out_bias=False,
        w_out_train=False,
        on_gpu=is_win,
    )

    print(
        f"Training RNN:",
        rnn,
        f"with dataset: [{orange}]{DATASET.name}",
        sep="\n",
    )

    # FIT
    loss_history = rnn.fit(
        DATASET(),
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        lr_milestones=lr_milestones,
        l2norm=0,
        stop_loss=stop_loss,
        plot_live=True,
    )

    plot_predictions(rnn, batch_size, DATASET)
    plot_training_loss(loss_history)
    plt.show()

    rnn.save(f"rnn_trained_with_{name}.pt")
