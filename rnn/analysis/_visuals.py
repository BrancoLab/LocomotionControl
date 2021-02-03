from vedo import Arrow, fitPlane, Plane, fitLine
import matplotlib.pyplot as plt
import numpy as np


from fcutils.plot.figure import clean_axes
from fcutils.plot.elements import plot_line_outlined

from myterial import (
    salmon,
    cyan,
    orange,
    red_dark,
    purple_dark,
    indigo_dark,
    green_dark,
    brown,
    blue_grey_dark,
)

from pyrnn._utils import npify

"""
    Visualization and plotting code for RNN analysis
"""
COLORS = dict(
    x=red_dark,
    y=purple_dark,
    r=red_dark,
    psy=purple_dark,
    v=indigo_dark,
    omega=cyan,
    theta=cyan,
    tau_R=orange,
    tau_L=salmon,
    nudot_R=orange,
    nudot_L=salmon,
    P=green_dark,
    N_R=brown,
    N_L=blue_grey_dark,
)


# ---------------------------------------------------------------------------- #
#                                   PLOTTING                                   #
# ---------------------------------------------------------------------------- #

# ---------------------------------- weights --------------------------------- #
def plot_rnn_weights(rnn):
    """
        Plots an RNN's weights matrices as heatmaps and histograms.

        Arguments:
            rnn: a pyrnn.RNN
    """
    W_in = npify(rnn.w_in.weight)
    W_rec = npify(rnn.w_rec.weight)
    W_out = npify(rnn.w_out.weight)

    f, axarr = plt.subplots(ncols=2, nrows=3, figsize=(16, 9))

    axarr[0, 0].imshow(W_in, cmap="bwr", aspect="equal")
    axarr[0, 0].set(title="$W_{in}$")
    axarr[0, 1].hist(W_in.ravel(), bins=64, density=True)
    axarr[0, 1].set(xlabel="weight value", ylabel="density")

    axarr[1, 0].imshow(W_rec, cmap="bwr", aspect="equal")
    axarr[1, 0].set(title="$W_{rec}$")
    axarr[1, 1].hist(W_rec.ravel(), bins=64, density=True)
    axarr[1, 1].set(xlabel="weight value", ylabel="density")

    axarr[2, 0].imshow(W_out, cmap="bwr", aspect="equal")
    axarr[2, 0].set(title="$W_{out}$")
    axarr[2, 1].hist(W_out.ravel(), bins=64, density=True)
    axarr[2, 1].set(xlabel="weight value", ylabel="density")

    return f


# ---------------------------------- RNN I/O --------------------------------- #


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


# ---------------------------------------------------------------------------- #
#                                    RENDER                                    #
# ---------------------------------------------------------------------------- #


def render_vectors(
    points, labels, colors, scale=1, showplane=False, showline=False
):
    """
        Creates a vedo Line from the origin along a vector to a point
        for each point in points (of unit length)

        Arguments:
            points: list of N points coordinates (3D)
            labels: list of N str with names for the vectors
            colors: list of N colors for the vectors
            scale: float. Use it to scale vectors lengths
            showplane: bool. If True a plane fitted to the vectors and origin is shown
            showline: bool. If True a line fitted to each point and the origin
    """
    actors = []
    for point, c, l in zip(points, colors, labels):
        actors.append(Arrow((0, 0, 0), point * scale, c=c, s=0.015).legend(l))

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
