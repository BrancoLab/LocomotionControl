import matplotlib.pyplot as plt
import os
from pyinspect.utils import timestamp
from pyrnn import RNN
from pyrnn.plot import plot_training_loss
from rich import print
import click
from loguru import logger
import json
import sys
from myterial import orange

from rnn.dataset.dataset import PredictNuDotFromXYT as DATASET
from rnn.dataset.dataset import is_win
from rnn.dataset import plot_predictions

# Set up
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

logger.add(
    sys.stderr,
    format="{time} {level} {message}",
    filter="my_module",
    level="INFO",
)

# ---------------------------------- Params ---------------------------------- #
MAKE_DATASET = False

n_units = 128

name = DATASET.name
batch_size = 64
epochs = 5000  # 300
lr_milestones = [500]
lr = 0.001
stop_loss = None

# ------------------------------- Fit/load RNN ------------------------------- #
@logger.catch
def make_rnn(data, winstor):
    logger.info("Creating RNN")
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
    logger.info(
        f"Rnn params:\n{json.dumps(rnn.params, sort_keys=True, indent=4)}"
    )
    return rnn


@logger.catch
def fit(rnn, winstor, data):
    print(
        f"Training RNN:", rnn, f"with dataset: [{orange}]{name}", sep="\n",
    )

    info = dict(
        dataset=data.name,
        dataset_length=len(data),
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        lr_milestones=lr_milestones,
        l2norm=0,
        stop_loss=stop_loss,
        plot_live=True,
        report_path=None,
    )
    logger.info(
        f"Training params:\n{json.dumps(info, sort_keys=True, indent=4)}"
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
        report_path=None,
    )
    print("Training finished, last loss: ", loss_history[-1])
    return loss_history


@logger.catch
def wrap_up(rnn, loss_history, winstor, data):
    logger.info("Wrapping up")

    NAME = f"rnn_trained_with_{name}.pt"
    f1 = plot_predictions(rnn, batch_size, DATASET)
    f2 = plot_training_loss(loss_history)

    if not winstor:
        plt.show()
        rnn.save(NAME)
    else:
        f1.savefig(data.rnn_folder / "predictions.png")
        f2.savefig(data.rnn_folder / "training_loss.png")

        rnn.save(str(data.rnn_folder / NAME))
        rnn.params_to_file(str(data.rnn_folder / f"rnn.txt"))

    logger.info(f"Saved RNN at: {NAME}")


@click.command()
@click.option("-w", "--winstor", is_flag=True, default=False)
def train(winstor):
    data = DATASET(dataset_length=10, winstor=winstor)

    if winstor:
        data.make_save_rnn_folder()
        logger.add(
            str(data.rnn_folder / "log_{time}.log"),
            backtrace=True,
            diagnose=True,
        )
    else:
        os.remove("out.log")
        logger.add("out.log", backtrace=True, diagnose=True)

    logger.info(
        "\n"
        + "#" * 60
        + "\n"
        + "  " * 5
        + f"Starting: {timestamp()}"
        + "     " * 5
        + "\n"
        + "#" * 60
        + "\n\n"
    )

    # Create RNN
    rnn = make_rnn(data, winstor)

    # fit
    loss_history = fit(rnn, winstor, data)

    # wrap up
    wrap_up(rnn, loss_history, winstor, data)


if __name__ == "__main__":
    if MAKE_DATASET:
        DATASET(truncate_at=None).make()
        DATASET(truncate_at=None).plot_random()
        DATASET().plot_durations()
    else:
        train()
