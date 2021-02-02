from loguru import logger
import json
import torch
from einops import repeat
import numpy as np


from fcutils.maths.signals import derivative
from fcutils.path import from_json, from_yaml

from pyrnn import RNN, is_win
from pyrnn.analysis import (
    FixedPoints,
    list_fixed_points,
)
from pyrnn._utils import torchify

from rnn.dataset import datasets

# ----------------------------------- misc ----------------------------------- #


def unpad(X, h, O, Y):
    """
        Sequences are padded with 0s during RNN training and inference.
        This function unpads the sequences for analysis

        Arguments:
            X: np.array of N_trials x N_samples x N inputs
            h: np.array of N_trials x N_samples x N units
            O: np.array of N_trials x N_samples x N outputs
            Y: np.array of N_trials x N_samples x N outputs

        Returns:
            X, h: same shapes but replacing the pads with np.nan
    """
    _X = X.copy()
    _h = h.copy()
    _O = O.copy()
    _Y = Y.copy()

    for trialn in range(X.shape[0]):
        try:
            stop = np.where(np.abs(derivative(X[trialn, :, 0])) > 0.1)[0][0]
        except IndexError:
            continue
        else:
            _X[trialn, stop:, :] = np.nan
            _h[trialn, stop:, :] = np.nan
            _O[trialn, stop:, :] = np.nan
            _Y[trialn, stop:, :] = np.nan

    return _X, _h, _O, _Y


# ------------------------------- Fixed points ------------------------------- #


def make_constant_inputs(rnn):
    """
        Makes a list of constant inputs (zeros) for a given RNN
    """
    constant_inputs = [
        repeat(
            torchify(np.zeros(rnn.input_size)).cuda(), "i -> b n i", b=1, n=1
        ),
    ]

    return constant_inputs


def fit_fps(rnn, h, fld, **kwargs):
    """
        Fit pyrnn FixedPoints analysis to identify fixed points in the dynamics

        Arguments:
            rnn: RNN class instance
            h: np.ndarray. (N trials, N frames, N units) array with hidden states
            fld: Path. Folder where the fixed points will be saved
            kwargs: paramaters to pass to `find_fixed_points`:
                n_initial_conditions=150,
                max_iters=9000,
                lr_decay_epoch=1500,
                max_fixed_points=n_fixed_points,
                gamma=0.1,

        Returns:
            fps: list of FixedPoint objects

    """
    logger.debug(f"Finding fixed points with h of shape {h.shape}")
    constant_inputs = make_constant_inputs(rnn)

    fp_finder = FixedPoints(rnn, speed_tol=1e-02, noise_scale=2)

    fp_finder.find_fixed_points(h, constant_inputs, **kwargs)

    # save number of fixed points
    fp_finder.save_fixed_points(fld / "fixed_point.json")

    # list fps
    fps = FixedPoints.load_fixed_points(fld / "fixed_point.json")
    logger.debug(f"Found {len(fps)} fixed points in total")
    list_fixed_points(fps)

    return fps


# ------------------------------------ I/O ----------------------------------- #


def get_file(folder, pattern):
    """ Finds the path of a file in a folder given a pattern """
    try:
        return list(folder.glob(pattern))[0]
    except IndexError:
        raise ValueError(
            f"Could not find file with pattern '{pattern}' in folder: '{folder.parent}/{folder.name}'"
        )


def load_from_folder(fld, winstor=False):
    """
        Loads a trained RNN from the folder with training outcomes
        Loads settings from rnn.yaml and uses that to load the correct
        dataset used for the RNN and to set the RNN params. 

        Arguments:
            fld: Path. Path to folder with RNN and metadata
            winstor: bool. True if the fikder lives on winstor

        Returns
            dataset: instance of DataSet subclass used for training
            RNN: instance of pyrnn.RNN loaded from saved model
            fps: inlist of FixedPoints loaded from file or empty list

    """
    logger.debug(f"Loading data from {fld.name}")

    # load params from yml or json
    try:
        settings_file = get_file(fld, "rnn.yaml")
        settings = json.loads(from_yaml(str(settings_file)))
    except Exception:
        settings_file = get_file(fld, "rnn.json")
        settings = from_json(settings_file)

        if isinstance(settings, str):
            settings = json.loads(settings)

    logger.debug("Loaded settings")

    # load dataset used for training
    dataset = datasets[settings["dataset_name"]](winstor=winstor)
    logger.debug(f'Loaded dataset: "{dataset.name}"')

    # set dataset settings for inference
    dataset.augment_probability = 0
    dataset.to_chunks = False
    dataset.warmup = False

    # load RNN
    del (
        settings["on_gpu"],
        settings["dataset_name"],
        settings["dataset_length"],
    )
    rnn = RNN.load(
        str(get_file(fld, "minloss.pt")),
        **settings,
        on_gpu=is_win,
        load_kwargs=dict(map_location=torch.device("cpu"))
        if not is_win
        else {},
    )
    logger.debug(f"Loaded RNN")

    # load fixed points
    try:
        fps_file = get_file(fld / "analysis", "fixed_point.json")
    except Exception:
        logger.debug("No fixed points data found when loading RNN from folder")
        fps = []
    else:
        fps = FixedPoints.load_fixed_points(fps_file)
        logger.debug(f"Loaded {len(fps)} fixed points from file")
    return dataset, rnn, fps
