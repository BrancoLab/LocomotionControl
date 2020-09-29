import matplotlib.pyplot as plt
import numpy as np
import logging

from fcutils.plotting.utils import clean_axes, save_figure
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.plot_elements import plot_line_outlined
from fcutils.maths.utils import derivative

from proj.utils.misc import load_results_from_folder
from proj.animation import variables_colors as colors


def _make_figure():
    f = plt.figure(figsize=(20, 8))

    gs = f.add_gridspec(2, 8)

    xy_ax = f.add_subplot(gs[:, :2])
    xy_ax.axis("equal")
    xy_ax.axis("off")

    tau_ax = f.add_subplot(gs[0, 2:4])
    sax = f.add_subplot(gs[1, 2:4])
    accel_ax = f.add_subplot(gs[0, 4:6])
    cost_ax = f.add_subplot(gs[1, 4:6])

    tau_int_ax = f.add_subplot(gs[0, 6:])
    acc_int_ax = f.add_subplot(gs[1, 6:])

    return f, xy_ax, tau_ax, sax, accel_ax, cost_ax, tau_int_ax, acc_int_ax


def _plot_xy(history, trajectory, plot_every, ax=None):
    # plot trajectory
    plot_line_outlined(
        ax,
        trajectory[:, 0],
        trajectory[:, 1],
        lw=1.5,
        color=colors["trajectory"],
        outline=0.5,
        outline_color="white",
    )
    ax.scatter(
        trajectory[::plot_every, 0],
        trajectory[::plot_every, 1],
        color="white",
        lw=1.5,
        edgecolors=colors["trajectory"],
        s=20,
        zorder=99,
    )

    # plot tracking
    plot_line_outlined(
        ax,
        history["x"],
        y=history["y"],
        color=colors["tracking"],
        lw=2,
        zorder=100,
        outline_color=[0.2, 0.2, 0.2],
    )


def _plot_control(history, ax=None):
    R, L = history["tau_r"], history["tau_l"]

    # plot traces
    plot_line_outlined(
        ax,
        R,
        color=colors["tau_r"],
        label="$\\tau_R$",
        lw=2,
        solid_capstyle="round",
    )
    plot_line_outlined(
        ax,
        L,
        color=colors["tau_l"],
        label="$\\tau_L$",
        lw=2,
        solid_capstyle="round",
    )
    ax.legend()


def _plot_v(history, trajectory, plot_every, ax=None):
    idx = 3
    v = history["v"]

    # plot traj speed
    ax.scatter(
        np.arange(len(trajectory[:, idx]))[::plot_every],
        trajectory[:, idx][::plot_every],
        color=desaturate_color(colors["v"]),
        label="trajectory speed",
        lw=1,
        edgecolors="white",
        s=50,
        alpha=0.5,
    )

    # plot history speed
    ax.plot(
        history["trajectory_idx"], v, color=colors["v"], lw=3, zorder=100,
    )


def _plot_accel(history, ax=None):
    v, omega = history["v"], history["omega"]
    vdot = derivative(v)
    omegadot = derivative(omega)

    plot_line_outlined(ax, vdot, lw=2, color=colors["v"], label="$\dot{v}$")
    plot_line_outlined(
        ax, omegadot, lw=2, color=colors["omega"], label="$\dot{\omega}$"
    )
    ax.legend()


def _plot_cost(cost_history, ax=None):
    for k in cost_history.columns:
        if "total" not in k:
            ax.plot(
                cost_history[k],
                label=k,
                lw=3,
                solid_capstyle="round",
                color=colors[k],
            )
    ax.legend()


def _plot_integrals(history, dt, tax=None, aax=None):
    R, L = history["nudot_right"], history["nudot_left"]

    plot_line_outlined(
        tax,
        R,
        color=desaturate_color(colors["tau_r"]),
        label="R_wheel_accel",
        lw=2,
        solid_capstyle="round",
    )
    plot_line_outlined(
        tax,
        L,
        color=desaturate_color(colors["tau_l"]),
        label="L_wheel_accel",
        lw=2,
        solid_capstyle="round",
    )
    tax.legend()

    # plot v and omega
    v, omega = history["v"], history["omega"]

    plot_line_outlined(
        aax,
        v,
        color=desaturate_color(colors["v"]),
        label="$v$",
        lw=2,
        solid_capstyle="round",
    )
    plot_line_outlined(
        aax,
        omega,
        color=desaturate_color(colors["omega"]),
        label="$\\omega$",
        lw=2,
        solid_capstyle="round",
    )
    aax.legend()


def plot_results(results_folder, plot_every=20, save_path=None):
    config, trajectory, history, cost_history = load_results_from_folder(
        results_folder
    )

    (
        f,
        xy_ax,
        tau_ax,
        sax,
        accel_ax,
        cost_ax,
        tau_int_ax,
        acc_int_ax,
    ) = _make_figure()

    _plot_xy(history, trajectory, plot_every, ax=xy_ax)
    _plot_control(history, ax=tau_ax)
    _plot_v(history, trajectory, plot_every, ax=sax)
    _plot_accel(history, ax=accel_ax)
    _plot_cost(cost_history, ax=cost_ax)
    _plot_integrals(history, config["dt"], tax=tau_int_ax, aax=acc_int_ax)

    clean_axes(f)

    if save_path is not None:
        save_figure(f, str(save_path))
        logging.info(f"Saved summary figure at: {save_path}")
