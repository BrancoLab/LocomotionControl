from loguru import logger
import json
import torch

from fcutils.file_io.io import load_yaml
from pyrnn import RNN, is_win
from pyrnn.analysis import (
    FixedPoints,
    list_fixed_points,
)
from rnn.dataset import datasets

# ------------------------------- Fixed points ------------------------------- #


def fit_fps(rnn, h, constant_inputs, fld, n_fixed_points=10):
    """
        Fit pyrnn FixedPoints analysis to identify fixed points in the dynamics

        Arguments:
            rnn: RNN class instance
            h: np.ndarray. (N trials, N frames, N units) array with hidden states
            constant_inputs: list. List of tensors with constant inputs for the analysis
            fld: Path. Folder where the fixed points will be saved
            n_fixed_points: int. Number of max fixed points to look for

    """
    fp_finder = FixedPoints(rnn, speed_tol=1e-02, noise_scale=2)

    fp_finder.find_fixed_points(
        h,
        constant_inputs,
        n_initial_conditions=150,
        max_iters=9000,
        lr_decay_epoch=1500,
        max_fixed_points=n_fixed_points,
        gamma=0.1,
    )

    fp_finder.save_fixed_points(fld / "3bit_fps.json")

    fps = FixedPoints.load_fixed_points(fld / "3bit_fps.json")
    list_fixed_points(fps)


# ------------------------------------ I/O ----------------------------------- #


def get_file(folder, pattern):
    """ Finds the path of a file in a folder given a pattern """
    try:
        return list(folder.glob(pattern))[0]
    except IndexError:
        raise ValueError(f"Could not find file in folder")


def load_from_folder(fld):
    """
        Loads a trained RNN from the folder with training outcomes
        Loads settings from rnn.yaml and uses that to load the correct
        dataset used for the RNN and to set the RNN params. 

        Arguments:
            fld: Path. Path to folder with RNN and metadata

        Returns
            dataset: instance of DataSet subclass used for training
            RNN: instance of pyrnn.RNN loaded from saved model

    """
    logger.debug(f"Loading data from {fld.name}")

    # load params from yml
    settings_file = get_file(fld, "rnn.yaml")
    settings = json.loads(load_yaml(str(settings_file)))
    logger.debug("Loaded settings")

    # load dataset used for training
    dataset = datasets[settings["dataset_name"]]()
    logger.debug(f'Loaded dataset: "{dataset.name}"')

    # load RNN
    del settings["on_gpu"], settings["dataset_name"]
    rnn = RNN.load(
        str(get_file(fld, "minloss.pt")),
        **settings,
        on_gpu=is_win,
        load_kwargs=dict(map_location=torch.device("cpu"))
        if not is_win
        else {},
    )
    logger.debug(f"Loaded RNN")

    return dataset, rnn
