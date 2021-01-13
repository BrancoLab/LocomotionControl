from loguru import logger
import json
import torch
from einops import repeat
import numpy as np
from vedo import Arrow, fitPlane, Plane, Tube
import matplotlib.pyplot as plt

from fcutils.file_io.io import load_yaml
from fcutils.plotting.utils import clean_axes
from fcutils.maths.utils import derivative

from myterial import salmon, blue_grey, teal

from pyrnn import RNN, is_win
from pyrnn.analysis import (
    FixedPoints,
    list_fixed_points,
)
from pyrnn._utils import torchify


from rnn.dataset import datasets

# ----------------------------------- misc ----------------------------------- #


def unpad(X, h):
    """
        Sequences are padded with 0s during RNN training and inference.
        This function unpads the sequences for analysis

        Arguments:
            X: np.array of N_trials x N_samples x N inputs
            h: np.array of N_trials x N_samples x N units

        Returns:
            X, h: same shapes but replacing the pads with np.nan
    """
    _X = X.copy()
    _h = h.copy()

    for trialn in range(X.shape[0]):
        try:
            stop = np.where(np.abs(derivative(X[trialn, :, 0])) > 0.1)[0][0]
        except IndexError:
            continue
        else:
            _X[trialn, stop:, :] = np.nan
            _h[trialn, stop:, :] = np.nan

    return _X, _h


# ------------------------------- visualization ------------------------------- #
def plot_inputs(X, labels):
    f, axarr = plt.subplots(ncols=len(labels), figsize=(16, 9))
    colors = (salmon, blue_grey, teal)

    for n, lab in enumerate(labels):
        for trialn in range(X.shape[0]):
            axarr[n].plot(X[trialn, :, n], color=colors[n], lw=1, alpha=0.8)

        axarr[n].set(xlabel="sim. frames", ylabel=lab)

    f.suptitle("Network inputs")
    clean_axes(f)

    return f


def render_inputs(X, labels):
    """
        Renders a set of inputs to the RNN in vedo to visualize
        alongsize the networks dynamics.

        X should be an b x n x N array with b=number of trials, n = number of time points and N<=3 is the number of inputs
    """

    # createa a tube for ecah trial with first two input variables and colored by the third
    actors = []
    for trialn in range(X.shape[0]):
        points = [
            (x, y, z)
            for x, y, z in zip(
                X[trialn, 1:, 0], X[trialn, 1:, 1], X[trialn, 1:, 2],
            )
        ]

        actors.append(Tube(points, r=0.01))

    # add custom axes with arrows
    for point, c, label in zip(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ("r", "g", "b"), labels
    ):
        actors.append(Arrow((0, 0, 0), point, c=c, s=0.005).legend(label))
    return actors


def render_vectors(points, labels, colors, showplane=False):
    """
        Creates a vedo Line from the origin along a vector to a point
        for each point in points (of unit length)
    """
    actors = [
        Arrow((0, 0, 0), point / np.linalg.norm(point), c=c, s=0.005).legend(l)
        for point, c, l in zip(points, colors, labels)
    ]

    if showplane:
        normal = fitPlane(np.array(points + [(0, 0, 0)])).normal
        actors.append(Plane(normal=normal, sx=5, sy=5, c="k", alpha=0.2))
    return actors


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


def fit_fps(rnn, h, fld, n_fixed_points=10):
    """
        Fit pyrnn FixedPoints analysis to identify fixed points in the dynamics

        Arguments:
            rnn: RNN class instance
            h: np.ndarray. (N trials, N frames, N units) array with hidden states
            fld: Path. Folder where the fixed points will be saved
            n_fixed_points: int. Number of max fixed points to look for

        Returns:
            fps: list of FixedPoint objects

    """
    logger.debug(
        f"Finding fixed pooints with h of shape {h.shape} and number of fixed points: {n_fixed_points}"
    )
    constant_inputs = make_constant_inputs(rnn)

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

    # save number of fixed points
    fp_finder.save_fixed_points(fld / "3bit_fps.json")

    # list fps
    fps = FixedPoints.load_fixed_points(fld / "3bit_fps.json")
    logger.debug(f"Found {len(fps)} fixed points in total")
    list_fixed_points(fps)

    return fps


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
