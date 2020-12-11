import matplotlib.pyplot as plt
import os
from pyinspect.utils import timestamp
from pyrnn import RNN
from pyrnn.plot import plot_training_loss
from rich import print
import click
from loguru import logger
import json
from myterial import orange

from rnn.dataset.dataset import PredictNuDotFromXYT as DATASET
from rnn.dataset.dataset import is_win
from rnn.dataset import plot_predictions

# Set up
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def setup_loggers(winstor, data):
    if winstor:
        main = str(data.rnn_folder / "log_{time}.log")
        train = str(data.rnn_folder / "training.log")
    else:
        main = "log.log"
        train = "training.log"

        os.remove("log.log")
        os.remove("training.log")

    logger.add(
        main,
        backtrace=True,
        diagnose=True,
        filter=lambda record: "main" in record["extra"],
        format="{time:YYYY-MM-DD at HH:mm} | {level} | {message}",
    )
    logger.add(
        train,
        filter=lambda record: "training" in record["extra"],
        format="{time:YYYY-MM-DD at HH:mm} |{level}| {message}",
    )
    logger.level("Params", no=38, color="<yellow>", icon="🐍")


# ---------------------------------- Params ---------------------------------- #
MAKE_DATASET = False
N_trials = 10

n_units = 128

name = DATASET.name
batch_size = 64
epochs = 50  # 300
lr_milestones = [500, 4000]
lr = 0.001
stop_loss = None

# --------------------------------- Make RNNR -------------------------------- #


@logger.catch
def make_rnn(data, winstor):
    logger.bind(main=True).info("Creating RNN")
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
    logger.bind(main=True).info(
        f"Rnn params:\n{json.dumps(rnn.params, sort_keys=True, indent=4)}",
    )
    return rnn


# ------------------------------------ fit ----------------------------------- #


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
    logger.bind(main=True).info(
        f"Training params:\n{json.dumps(info, sort_keys=True, indent=4)}",
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
        logger=logger,
    )
    print("Training finished, last loss: ", loss_history[-1])
    return loss_history


# ---------------------------------- wrap up --------------------------------- #


@logger.catch
def wrap_up(rnn, loss_history, winstor, data):
    logger.bind(main=True).info("Wrapping up")

    # save RNN
    NAME = f"rnn_trained_with_{name}.pt"
    if winstor:
        NAME = str(data.rnn_folder / NAME)
        rnn.params_to_file(str(data.rnn_folder / f"rnn.txt"), overwrite=True)
    rnn.save(NAME, overwrite=True)

    # make/save plots
    f1 = plot_predictions(rnn, batch_size, data)
    f2 = plot_training_loss(loss_history)

    if not winstor:
        plt.show()
    else:
        f1.savefig(data.rnn_folder / "predictions.png")
        f2.savefig(data.rnn_folder / "training_loss.png")

    logger.bind(main=True).info(f"Saved RNN at: {NAME}")


# ---------------------------------------------------------------------------- #
#                                   MAIN FUNC                                  #
# ---------------------------------------------------------------------------- #


@click.command()
@click.option("-w", "--winstor", is_flag=True, default=False)
def train(winstor):
    data = DATASET(dataset_length=N_trials, winstor=winstor)

    if winstor:
        data.make_save_rnn_folder()

    setup_loggers(winstor, data)
    logger.bind(main=True).info(
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
