import matplotlib.pyplot as plt
import numpy as np

from fcutils.plotting.utils import clean_axes, save_figure
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.plot_elements import plot_line_outlined
from fcutils.maths.utils import derivative

from proj.utils import load_results_from_folder, seagreen


def _make_figure():
    f = plt.figure(figsize=(16, 8))

    gs = f.add_gridspec(2, 6)

    xy_ax = f.add_subplot(gs[:, :2])
    xy_ax.axis("equal")
    xy_ax.axis("off")

    tau_ax = f.add_subplot(gs[0, 2:4])
    sax = f.add_subplot(gs[1, 2:4])
    accel_ax = f.add_subplot(gs[0, 4:])
    cost_ax = f.add_subplot(gs[1, 4:])

    return f, xy_ax, tau_ax, sax, accel_ax, cost_ax


def _plot_xy(history, trajectory, plot_every, ax=None):
    # plot trajectory
    ax.scatter(
        trajectory[::plot_every, 0],
        trajectory[::plot_every, 1],
        s=50,
        color=[0.4, 0.4, 0.4],
        lw=1,
        edgecolors="white",
    )

    # plot tracking
    ax.plot(
        history["x"],
        history["y"],
        lw=9,
        color=desaturate_color(seagreen),
        zorder=-1,
        solid_capstyle="round",
    )
    plot_line_outlined(
        ax,
        history["x"],
        y=history["y"],
        color=desaturate_color(seagreen),
        lw=12,
        zorder=-1,
    )


def _plot_control(history, ax=None):
    R, L = history["tau_r"], history["tau_l"]

    # plot traces
    plot_line_outlined(
        ax,
        R,
        color="r",
        label="$\\tau_R$",
        lw=3,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    plot_line_outlined(
        ax,
        L,
        color="b",
        label="$\\tau_L$",
        lw=3,
        solid_joinstyle="round",
        solid_capstyle="round",
    )
    ax.legend()


def _plot_v(history, trajectory, plot_every, ax=None):
    color = "#B22222"
    idx = 3
    v = history["v"]

    # plot traj speed
    ax.scatter(
        np.arange(len(trajectory[:, idx]))[::plot_every],
        trajectory[:, idx][::plot_every],
        color=color,
        label="trajectory speed",
        lw=1,
        edgecolors="white",
        s=100,
    )

    # plot history speed
    ax.plot(
        v, color=desaturate_color("m"), lw=9, zorder=-1,
    )


def _plot_accel(history, ax=None):
    v, omega = history["v"], history["omega"]
    vdot = derivative(v)
    omegadot = derivative(omega)

    ax.plot(vdot, lw=3, color="m", label="$v$")
    ax.plot(omegadot, lw=3, color="g", label="$\omega$")
    ax.legend()


def plot_results(results_folder, plot_every=20, save_path=None):
    config, trajectory, history = load_results_from_folder(results_folder)

    f, xy_ax, tau_ax, sax, accel_ax, cost_ax = _make_figure()

    _plot_xy(history, trajectory, plot_every, ax=xy_ax)
    _plot_control(history, ax=tau_ax)
    _plot_v(history, trajectory, plot_every, ax=sax)
    _plot_accel(history, ax=accel_ax)

    clean_axes(f)

    if save_path is not None:
        save_figure(f, str(save_path))
