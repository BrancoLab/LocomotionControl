import matplotlib.pyplot as plt
import os
from pyrnn import RNN
from pyrnn.plot import plot_training_loss
from rich import print
from myterial import orange
import click

from rnn.dataset.dataset import PredictNuDotFromXYT as DATASET
from rnn.dataset.dataset import is_win
from rnn.dataset import plot_predictions

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

MAKE_DATASET = False

# ---------------------------------- Params ---------------------------------- #
n_units = 128

name = DATASET.name
batch_size = 64
epochs = 5000  # 300
lr_milestones = [500]
lr = 0.001
stop_loss = None


# ------------------------------- Fit/load RNN ------------------------------- #
@click.command()
@click.option("-w", "--winstor", is_flag=True, default=False)
def train(winstor):
    data = DATASET(dataset_length=10, winstor=winstor)

    if winstor:
        data.make_save_rnn_folder()

    # Create RNN
    rnn = RNN(
        input_size=len(data.inputs_names),
        output_size=len(data.outputs_names),
        n_units=n_units,
        dale_ratio=None,
        autopses=True,
        w_in_bias=False,
        w_in_train=False,
        w_out_bias=False,
        w_out_train=False,
        on_gpu=is_win if not winstor else True,
    )

    print(
        f"Training RNN:", rnn, f"with dataset: [{orange}]{name}", sep="\n",
    )

    # FIT
    loss_history = rnn.fit(
        data,
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        lr_milestones=lr_milestones,
        l2norm=0,
        stop_loss=stop_loss,
        plot_live=True,
        report_path=str(data.rnn_folder / f"report.txt") if winstor else None,
    )

    if not winstor:
        plot_predictions(rnn, batch_size, DATASET)
        plot_training_loss(loss_history)
        plt.show()

        rnn.save(f"rnn_trained_with_{name}.pt")
    else:
        rnn.save(data.rnn_folder / f"rnn_trained_with_{name}.pt")
        rnn.params_to_file(data.rnn_folder / f"rnn.txt")


if __name__ == "__main__":
    if MAKE_DATASET:
        DATASET(truncate_at=None).make()
        DATASET(truncate_at=None).plot_random()
        DATASET().plot_durations()
    else:
        train()
