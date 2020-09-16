import matplotlib.pyplot as plt
import numpy as np

from proj.utils import polar_to_cartesian


def plot_trajectory_polar(traj):

    f = plt.figure()
    ax = f.add_subplot(121)
    pax = f.add_subplot(122, projection="polar")

    # plot in cartesian coordinates
    x, y = polar_to_cartesian(traj[:, 0], traj[:, 1])
    ax.scatter(x, y, cmap="bwr", c=np.arange(len(x)))

    # plot in polar coordinates
    pax.scatter(
        traj[:, 1], traj[:, 0], cmap="bwr", c=np.arange(len(traj[:, 0]))
    )


def plot_trajectory(traj):
    plt.ioff()

    f, axarr = plt.subplots(
        figsize=(10, 16), nrows=2, sharex=True, sharey=True
    )

    sc = axarr[0].scatter(
        traj[:, 0],
        traj[:, 1],
        c=traj[:, 2],
        lw=1,
        edgecolors="k",
        label="$\omega$",
    )
    axarr[0].set(
        title="$\omega$",
        xlabel="X",
        ylabel="Y",
        xlim=[traj[:, 0].min() - 10, traj[:, 0].max() + 10],
        ylim=[traj[:, 1].min() - 10, traj[:, 1].max() + 10],
    )
    # # axarr[0].axis("equal")
    # # axarr[0].legend()
    plt.colorbar(sc, ax=axarr[0])

    sc2 = axarr[1].scatter(
        traj[:, 0], traj[:, 1], c=traj[:, 3], lw=1, label="$v$"
    )
    axarr[1].set(
        title="v",
        xlabel="X",
        ylabel="Y",
        xlim=[traj[:, 0].min() - 10, traj[:, 0].max() + 10],
        ylim=[traj[:, 1].min() - 10, traj[:, 1].max() + 10],
    )
    # axarr[1].axis("equal")
    # axarr[1].legend()
    plt.colorbar(sc2, ax=axarr[1])

    plt.show()
