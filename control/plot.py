import matplotlib.pyplot as plt
import numpy as np
import logging

from fcutils.plotting.utils import clean_axes, save_figure
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.plot_elements import plot_line_outlined
from fcutils.video.utils import (
    get_cap_from_images_folder,
    save_videocap_to_video,
)
from fcutils.maths.utils import rolling_mean

from .history import load_results_from_folder

# from .config import dt

colors = dict(
    x=[0.5, 0.5, 0.5],
    y=[0.2, 0.2, 0.2],
    theta="#228B22",
    v="#8B008B",
    omega="#FA8072",
    tau_l="#87CEEB",
    tau_r="#FF4500",
    trajectory=[0.6, 0.6, 0.6],
    tracking="#2E8B57",
    P="#008060",
    N_r="#ff8c00",
    N_l="#e2734e",
)


def animate_from_images(folder, savepath, fps):
    cap = get_cap_from_images_folder(folder, img_format="%2d.png")
    save_videocap_to_video(cap, savepath, ".mp4", fps=fps)

    gifpath = savepath.replace(".mp4", ".gif")
    logging.info(
        "To save the video as GIF, use: \n"
        + f'ffmpeg -i "{savepath}" -f gif "{gifpath}"'
    )


def _make_figure():
    f = plt.figure(figsize=(20, 12))

    gs = f.add_gridspec(3, 6)

    xy_ax = f.add_subplot(gs[:2, :2])
    xy_ax.axis("equal")
    # xy_ax.axis("off")

    control_ax = f.add_subplot(gs[0, 2:4])
    sax = f.add_subplot(gs[2, :2])  # speed trajectory
    # accel_ax = f.add_subplot(gs[0, 4:6])
    # cost_ax = f.add_subplot(gs[1, 4:6])

    tau_ax = f.add_subplot(gs[0, 4:6])
    omega_ax = f.add_subplot(gs[1, 4:6])
    theta_ax = f.add_subplot(gs[1, 2:4])

    return f, xy_ax, control_ax, sax, tau_ax, omega_ax, theta_ax


def _plot_xy(history, trajectory, plot_every, duration, ax=None):
    # plot trajectory
    plot_line_outlined(
        ax,
        trajectory[:, 0],
        trajectory[:, 1],
        lw=2.5,
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

    ax.scatter(
        history["x"].values[0],
        history["y"].values[0],
        color=colors["tracking"],
        lw=2,
        zorder=120,
        edgecolors=[0.2, 0.2, 0.2],
        s=200,
    )

    # Set ax properties
    ax.set(xlabel="cm", ylabel="cm", title=f"Duration: {duration}s")


def _plot_control(history, ax=None):
    P, R, L = history["P"], history["N_r"], history["N_l"]

    for name, var in zip(("P", "N_r", "N_l"), (P, R, L)):
        ax.plot(
            var,
            color=colors[name],
            label=f"${name}$",
            lw=4,
            solid_capstyle="round",
            alpha=0.5,
        )

        plot_line_outlined(
            ax,
            rolling_mean(var, 21),
            color=colors[name],
            lw=2,
            solid_capstyle="round",
            alpha=1,
        )

    ax.legend()
    ax.set(
        xlabel="# frames",
        ylabel="Torque\n($\\frac{cm^2 g}{s^2}$)",
        title="Control history",
    )


def _plot_tau(history, ax=None):
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
    ax.set(
        xlabel="# frames",
        ylabel="Torque\n($\\frac{cm^2 g}{s^2}$)",
        title="Wheel torques",
    )


def _plot_v(history, trajectory, plot_every, ax=None):
    # plot traj speed
    ax.scatter(
        np.arange(len(trajectory[:, 3]))[::plot_every],
        trajectory[:, 3][::plot_every],
        color=desaturate_color(colors["v"]),
        label="trajectory speed",
        lw=1,
        edgecolors="white",
        s=50,
        alpha=0.5,
    )

    # plot history speed
    ax.plot(
        history["trajectory_idx"],
        history["v"],
        color=colors["v"],
        lw=3,
        zorder=100,
    )

    ax.set(
        xlabel="Trajectory idx",
        ylabel="Speed (cm/s)",
        title="Speed trajectory",
    )


def _plot_omega(history, trajectory, plot_every, ax=None):
    # plot traj speed
    ax.scatter(
        np.arange(len(trajectory[:, 4]))[::plot_every],
        np.degrees(trajectory[:, 4][::plot_every]),
        color=desaturate_color(colors["omega"]),
        label="trajectory speed",
        lw=1,
        edgecolors="white",
        s=50,
        alpha=0.5,
    )

    # plot history speed
    ax.plot(
        history["trajectory_idx"],
        np.degrees(history["omega"]),
        color=colors["omega"],
        lw=3,
        zorder=100,
    )

    ax.set(
        xlabel="Trajectory idx",
        ylabel="Ang. speed. (deg/s)",
        title="Ang. speed. trajectory",
    )


def _plot_theta(history, trajectory, plot_every, ax=None):
    # plot traj speed
    ax.scatter(
        np.arange(len(trajectory[:, 2]))[::plot_every],
        np.degrees(trajectory[:, 2][::plot_every]),
        color=desaturate_color(colors["theta"]),
        label="trajectory orientation",
        lw=1,
        edgecolors="white",
        s=50,
        alpha=0.5,
    )

    # plot history speed
    ax.plot(
        history["trajectory_idx"],
        np.degrees(history["theta"]),
        color=colors["theta"],
        lw=3,
        zorder=100,
    )

    ax.set(
        xlabel="Trajectory idx",
        ylabel="Theta (deg)",
        title="Theta trajectory",
    )


def plot_results(results_folder, plot_every=20, save_path=None):
    history, info, trajectory, trial = load_results_from_folder(results_folder)
    duration = info["duration"]

    f, xy_ax, control_ax, sax, tau_ax, omega_ax, theta_ax = _make_figure()

    _plot_xy(history, trajectory, plot_every, duration, ax=xy_ax)
    _plot_control(history, ax=control_ax)
    _plot_tau(history, ax=tau_ax)
    _plot_v(history, trajectory, plot_every, ax=sax)
    _plot_omega(history, trajectory, plot_every, ax=omega_ax)
    _plot_theta(history, trajectory, plot_every, ax=theta_ax)

    clean_axes(f)
    f.tight_layout()

    if save_path is not None:
        save_figure(f, str(save_path))
        logging.info(f"Saved summary figure at: {save_path}")
