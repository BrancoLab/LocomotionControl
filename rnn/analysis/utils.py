from loguru import logger
import json
import torch
from einops import repeat
import numpy as np
from vedo import Arrow, fitPlane, Plane, fitLine
import matplotlib.pyplot as plt

from fcutils.file_io.io import load_yaml
from fcutils.plotting.utils import clean_axes
from fcutils.plotting.plot_elements import plot_line_outlined
from fcutils.maths.utils import derivative

from myterial import salmon, purple, indigo, cyan, orange

from pyrnn import RNN, is_win
from pyrnn.analysis import (
    FixedPoints,
    list_fixed_points,
)
from pyrnn._utils import torchify


from rnn.dataset import datasets


COLORS = dict(x=purple, y=indigo, theta=cyan, tau_R=orange, tau_L=salmon,)


# ----------------------------------- misc ----------------------------------- #


def unpad(X, h, O):
    """
        Sequences are padded with 0s during RNN training and inference.
        This function unpads the sequences for analysis

        Arguments:
            X: np.array of N_trials x N_samples x N inputs
            h: np.array of N_trials x N_samples x N units
            h: np.array of N_trials x N_samples x N outputs

        Returns:
            X, h: same shapes but replacing the pads with np.nan
    """
    _X = X.copy()
    _h = h.copy()
    _O = O.copy()

    for trialn in range(X.shape[0]):
        try:
            stop = np.where(np.abs(derivative(X[trialn, :, 0])) > 0.1)[0][0]
        except IndexError:
            continue
        else:
            _X[trialn, stop:, :] = np.nan
            _h[trialn, stop:, :] = np.nan
            _O[trialn, stop:, :] = np.nan

    return _X, _h, _O


# ------------------------------- visualization ------------------------------- #
def plot_inputs(X, labels):
    """
        Plot a network's inputs across trials

        X should be an b x n x N array with b=number of trials, n = number of time points and N<=3 is the number of inputs
        lables: list of str with name for each column in the last dimension of X

    """
    f, axarr = plt.subplots(ncols=len(labels), figsize=(16, 9))
    for n, lab in enumerate(labels):
        for trialn in range(X.shape[0]):
            axarr[n].plot(X[trialn, :, n], color=COLORS[lab], lw=1, alpha=0.5)

        plot_line_outlined(
            axarr[n],
            np.nanmean(X[:, :, n], 0),
            color=salmon,
            outline=2,
            lw=5,
            alpha=1,
            zorder=100,
        )

        axarr[n].set(xlabel="sim. frames", ylabel=lab)

    f.suptitle("Network INPUTs")
    clean_axes(f)

    return f


def plot_outputs(O, labels):
    """
        Plot a network's outputs across trials

        Arguments:
            O: np.array with TxSxO shape
            labels: list of str of O length with output's names
    """

    f, axarr = plt.subplots(ncols=len(labels), figsize=(16, 9))

    for n, lab in enumerate(labels):
        for trialn in range(O.shape[0]):
            axarr[n].plot(O[trialn, :, n], color=COLORS[lab], lw=1, alpha=0.5)

        plot_line_outlined(
            axarr[n],
            np.nanmean(O[:, :, n], 0),
            color=salmon,
            outline=2,
            lw=5,
            alpha=1,
            zorder=100,
        )

        axarr[n].set(xlabel="sim. frames", ylabel=lab)

    f.suptitle("Network OUTPUTs")
    clean_axes(f)

    return f


def render_vectors(points, labels, colors, showplane=False, showline=False):
    """
        Creates a vedo Line from the origin along a vector to a point
        for each point in points (of unit length)

        Arguments:
            points: list of N points coordinates (3D)
            labels: list of N str with names for the vectors
            colors: list of N colors for the vectors
            showplane: bool. If True a plane fitted to the vectors and origin is shown
            showline: bool. If True a line fitted to each point and the origin
    """
    actors = []
    for point, c, l in zip(points, colors, labels):
        actors.append(
            Arrow(
                (0, 0, 0), (point / np.linalg.norm(point)) * 2, c=c, s=0.015
            ).legend(l)
        )

        if showline:
            actors.append(
                fitLine(np.array(list(point) + [0, 0, 0]))
                .c(c)
                .alpha(0.4)
                .lw(0.01)
            )

    if showplane:
        normal = fitPlane(np.array(points + [(0, 0, 0)])).normal
        actors.append(Plane(normal=normal, sx=10, sy=10, c="k", alpha=0.3))

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

    # set dataset settings for inference
    dataset.augment_probability = 0
    dataset.to_chunks = False
    dataset.warmup = False

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
