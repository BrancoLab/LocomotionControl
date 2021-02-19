# %%
from loguru import logger
import pandas as pd
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter

# from sklearn.metrics import mean_squared_error as MSE


from myterial import (
    blue_darker,
    orange_darker,
    blue,
    orange,
)

from fcutils import path
from fcutils.plot.figure import clean_axes, save_figure
from fcutils.plot.elements import plot_line_outlined
from fcutils.progress import track
from fcutils.maths.array import find_nearest
from fcutils.maths import rolling_mean

module_path = Path(os.path.abspath(os.path.join("."))).parent
sys.path.append(str(module_path))
sys.path.append("./")
from experimental_validation.trials import Trials
from experimental_validation import paths
from control.utils import get_theta_from_xy

# %%
# Get simulations and lookup
simulations_folder = Path(
    "Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD\\SIMULATIONS"
)

simulations_lookup = {}
for sim in path.subdirs(simulations_folder):
    try:
        trial = pd.read_hdf(sim / "trial.h5")
    except FileNotFoundError:
        logger.warning(f"Did not find trial file for {sim}")
        continue
    simulations_lookup[trial["name"]] = sim
logger.info(f"Loaded simulations for {len(simulations_lookup)} trials")

trials = Trials(only_tracked=True)

BAD_TRIALS = (  # trials in which the simulation went wrong
    0,
    8,
    22,
    24,
    49,
    51,
)

# %%
"""
    For each trial, plot the real trial DATA and the results from the simulation
"""
sos_th = 1
sos = butter(6, sos_th, output="sos", fs=60)

mse_records = []
for trialn in track(range(len(trials)), total=len(trials)):
    # load all data
    trial = trials[trialn]
    if not trial.good or len(trial) < 50:
        continue

    try:
        simulated = pd.read_hdf(
            simulations_lookup[trial.name] / "trial.h5"
        )  # trial for simulation
    except KeyError:
        logger.warning(
            f"Skipping {trial.name} [{trialn}] as no subfolder was found"
        )
        continue

    simulation = pd.read_hdf(
        simulations_lookup[trial.name] / "history.h5"
    )  # simulation results
    simulation["dirmvmt"] = get_theta_from_xy(simulation.x, simulation.y)
    simulation.dirmvmt[0] = simulation.dirmvmt[1].copy()
    simulation.dirmvmt = simulation.dirmvmt

    assert trial.name == simulated["name"], "loaded the wrong one mate"

    # get sim start wrt to trial start
    sim_start_idx = find_nearest(trial.body.y, simulation.y[0])

    # get trial frame -> time and sim frame -> time
    trial_dur = len(trial) / 60  # 60 is the fps of trial recordings
    trial_f2t = np.linspace(0, trial_dur, len(trial))

    sim_dur = len(simulation.x) / 200  # fps of simulation
    sim_start = (sim_start_idx * sim_dur) / len(
        trial
    ) + 0.075  # 0.075 from lookahead
    sim_f2t = np.linspace(sim_start, sim_dur + sim_start, len(simulation.x))

    xaxprop = dict(
        xlim=[0, 50],
        xticks=[0, 50, 100],
        xticklabels=[0, round(trial_dur / 2, 3), round(trial_dur, 3)],
    )

    # get ratio between trial and simulation speed
    trial_speed = np.array(
        [trial.v[find_nearest(trial_f2t, t)] for t in sim_f2t]
    )
    speed_ratio = rolling_mean(trial_speed, 60) / simulation.v

    # get avg paw speed on each side
    RIGHT = (trial.right_hl.speed + trial.right_fl.speed) / 2
    LEFT = (trial.left_hl.speed + trial.left_fl.speed) / 2

    # interpolate time series to make them have the same length
    maxl = 50
    data = dict(
        left_raw=rolling_mean(trial.left_hl.speed, 5),
        right_raw=rolling_mean(trial.right_hl.speed, 5),
        left=rolling_mean(LEFT, 20),
        right=rolling_mean(RIGHT, 20),
        phidot_left=rolling_mean(simulation.phidot_l * speed_ratio, 20),
        phidot_right=rolling_mean(simulation.phidot_r * speed_ratio, 20),
        trial_speed=rolling_mean(trial.v, 20),
        simulation_speed=rolling_mean(simulation.v, 20),
    )

    data = {
        k: np.interp(np.linspace(0, 1, maxl), np.linspace(0, 1, len(v)), v)
        for k, v in data.items()
    }

    # calc MSE
    # mse_records.append(
    #     MSE(
    #         np.vstack([simulation.phidot_l * speed_ratio, simulation.phidot_r * speed_ratio]),
    #         np.vstack([LEFT, RIGHT]),
    #     )
    # )

    # plot
    f, axarr = plt.subplots(figsize=(16, 9), nrows=2, ncols=2)
    axarr = axarr.flatten()

    # plot XY
    axarr[0].plot(
        simulated.x,
        simulated.y,
        lw=8,
        color="k",
        label="trajectory",
        alpha=0.6,
        zorder=-1,
    )
    axarr[0].scatter(
        simulation.x[::20],
        simulation.y[::20],
        s=120,
        color=[0.2, 0.2, 0.2],
        zorder=2,
    )
    axarr[0].scatter(
        simulation.x[::20], simulation.y[::20], s=100, color="salmon", zorder=3
    )
    axarr[0].scatter(
        simulation.x[0],
        simulation.y[0],
        s=400,
        lw=2,
        edgecolors=[0.2, 0.2, 0.2],
        color="salmon",
        alpha=1,
        label="simulated",
        zorder=4,
    )

    axarr[0].axis("equal")
    axarr[0].set(xlabel="X (cm)", ylabel="Y (cm)")
    axarr[0].legend()

    # plot speed
    axarr[2].plot(
        data["trial_speed"], lw=8, alpha=0.6, color="k", label="real",
    )
    plot_line_outlined(
        axarr[2],
        data["simulation_speed"],
        lw=6,
        color="salmon",
        label="simulation",
    )
    axarr[2].set(xlabel="time (s)", ylabel="$v$", **xaxprop)
    axarr[2].legend()

    # plot paw speeds raw
    axarr[1].plot(
        data["right_raw"], lw=8, zorder=-1, color=orange, label="$RIGHT$",
    )
    axarr[1].plot(
        data["left_raw"], lw=8, zorder=-1, color=blue, label="$LEFT$",
    )

    plot_line_outlined(
        axarr[1],
        data["phidot_right"],
        lw=6,
        color=orange_darker,
        label="$\phi_R$",
        outline=2,
    )
    plot_line_outlined(
        axarr[1],
        data["phidot_left"],
        lw=6,
        color=blue_darker,
        label="$\phi_L$",
        outline=2,
    )
    axarr[1].legend()
    axarr[1].set(
        xlabel="time (s)", ylabel="speed (cm/s)$", **xaxprop, ylim=[0, 60]
    )

    # plot paw speeds processed
    axarr[3].plot(
        data["right"], lw=8, zorder=-1, color=orange, label="$RIGHT$",
    )
    axarr[3].plot(
        data["left"], lw=8, zorder=-1, color=blue, label="$LEFT$",
    )
    plot_line_outlined(
        axarr[3],
        data["phidot_right"],
        lw=6,
        color=orange_darker,
        label="$\phi_R$",
        outline=2,
    )
    plot_line_outlined(
        axarr[3],
        data["phidot_left"],
        lw=6,
        color=blue_darker,
        label="$\phi_L$",
        outline=2,
    )
    axarr[3].legend()
    axarr[3].set(
        xlabel="time (s)", ylabel="speed (cm/s)$", **xaxprop, ylim=[0, 60]
    )

    clean_axes(f)

    # save figure
    plt.show()
    break
    save_figure(
        f,
        paths.folder_2WDD
        / "ANALYSIS"
        / "trials"
        / f"sim2real_trial_{trialn}.png",
        close=True,
    )
    # break


# %%
# interpolate time series to make them have the same length
maxl = 100
data = dict(
    left_raw=rolling_mean(trial.left_hl.speed, 5),
    right_raw=rolling_mean(trial.right_hl.speed, 5),
    left=rolling_mean(LEFT, 20),
    right=rolling_mean(RIGHT, 20),
    phidot_left=rolling_mean(simulation.phidot_l * speed_ratio, 20),
    phidot_right=rolling_mean(simulation.phidot_r * speed_ratio, 20),
)

iterp_data = {
    k: np.interp(np.linspace(0, 1, maxl), np.linspace(0, 1, len(v)), v)
    for k, v in data.items()
}

f, axarr = plt.subplots(nrows=2)
for key in data.keys():
    axarr[0].plot(data[key], label=key)
    axarr[1].plot(iterp_data[key])
axarr[0].legend()
# %%
