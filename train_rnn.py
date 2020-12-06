import matplotlib.pyplot as plt
import os
from pyrnn import RNN
from pyrnn.plot import plot_training_loss
from rich import print
from myterial import orange

from rnn.dataset.dataset import PredictTauFromXYTVO as DATASET
from rnn.dataset import plot_predictions

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

MAKE_DATASET = False

# ---------------------------- Preprocess dataset ---------------------------- #
if MAKE_DATASET:
    DATASET(truncate_at=None).make()
    DATASET(truncate_at=None).plot_random()

# ---------------------------------- Params ---------------------------------- #
n_units = 256

name = DATASET.name
batch_size = 64
epochs = 150  # 300
lr_milestones = [150, 450]
lr = 0.01
stop_loss = 0.002

# ------------------------------- Fit/load RNN ------------------------------- #
if not MAKE_DATASET:
    # Create RNN
    rnn = RNN(
        input_size=5,
        output_size=2,
        n_units=n_units,
        dale_ratio=None,
        autopses=True,
        w_in_bias=True,
        w_in_train=True,
        w_out_bias=True,
        w_out_train=True,
        on_gpu=False,
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
        plot_live=False,
    )
    rnn.save(f"{name}.pt")

    plot_predictions(rnn, batch_size, DATASET)
    plot_training_loss(loss_history)
    plt.show()
