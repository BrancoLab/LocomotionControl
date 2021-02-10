# %%
from loguru import logger
import pandas as pd
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter


from myterial import (
    green_darker,
    indigo_dark,
    blue_darker,
    orange_darker,
    blue,
    orange,
)

from fcutils import path
from fcutils.plot.figure import clean_axes, save_figure
from fcutils.progress import track
from fcutils.maths.array import find_nearest
from fcutils.maths import rolling_mean

module_path = Path(os.path.abspath(os.path.join("."))).parent
sys.path.append(str(module_path))
sys.path.append("./")
from experimental_validation.trials import Trials
from experimental_validation import paths
from experimental_validation._tracking import unwrap
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

# %%
"""
    For each trial, plot the real trial DATA and the results from the simulation
"""
sos_th = 1
sos = butter(6, sos_th, output="sos", fs=60)

for trialn in track(range(len(trials)), total=len(trials)):
    # if trialn != 15:
    #     continue

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
    tlim = [sim_start, sim_dur + sim_start]

    # get ratio between trial and simulation speed
    trial_speed = np.array(
        [trial.v[find_nearest(trial_f2t, t)] for t in sim_f2t]
    )
    speed_ratio = rolling_mean(trial_speed, 60) / simulation.v

    # plot
    f, axarr = plt.subplots(figsize=(10, 22), nrows=4)

    # plot XY
    axarr[0].scatter(
        trial.body.x,
        trial.body.y,
        s=200,
        lw=2,
        edgecolors=[0.2, 0.2, 0.2],
        color=[0.8, 0.8, 0.8],
        label="real",
    )
    axarr[0].scatter(
        simulation.x,
        simulation.y,
        s=100,
        lw=0.5,
        edgecolors=[0.2, 0.2, 0.2],
        color="salmon",
        alpha=1,
        label="simulated",
    )
    axarr[0].scatter(
        simulation.x[0],
        simulation.y[0],
        s=400,
        lw=2,
        edgecolors=[0.2, 0.2, 0.2],
        color="salmon",
        alpha=1,
    )
    axarr[0].plot(
        simulated.x,
        simulated.y,
        lw=4,
        color="k",
        label="sim. trajectory",
        zorder=-1,
    )
    axarr[0].set(xlabel="X (cm)", ylabel="Y (cm)")
    axarr[0].legend()

    # plot theta
    axarr[1].plot(
        trial_f2t,
        rolling_mean(unwrap(trial.dirmvmt), 12),
        lw=6,
        color=green_darker,
        label="real",
    )
    axarr[1].plot(
        sim_f2t,
        unwrap(simulation.dirmvmt),
        lw=4,
        color="salmon",
        label="simulation",
    )
    axarr[1].set(xlabel="time (s)", ylabel="$\\theta$", xlim=tlim)
    axarr[1].legend()

    # plot speed
    axarr[2].plot(
        trial_f2t,
        rolling_mean(trial.v, 12),
        lw=6,
        color=indigo_dark,
        label="real",
    )
    axarr[2].plot(
        sim_f2t, simulation.v, lw=4, color="salmon", label="simulation"
    )
    axarr[2].set(xlabel="time (s)", ylabel="$v$", xlim=tlim)
    axarr[2].legend()

    # plot wheel and paw speeds
    # axarr[3].plot(
    #     trial_f2t,
    #     sosfiltfilt(sos, trial.right_hl.speed),
    #     lw=4,
    #     color=orange,
    #     label="$RH$",
    # )
    # axarr[3].plot(
    #     trial_f2t,
    #     sosfiltfilt(sos, trial.left_hl.speed),
    #     lw=4,
    #     color=blue,
    #     label="$LH$",
    # )
    axarr[3].plot(
        trial_f2t,
        rolling_mean((trial.right_hl.speed + trial.right_fl.speed) / 2, 20),
        lw=4,
        # ls='--',
        color=orange,
        label="$RIGHT$",
    )
    axarr[3].plot(
        trial_f2t,
        rolling_mean((trial.left_hl.speed + trial.left_fl.speed) / 2, 20),
        lw=4,
        # ls='--',
        color=blue,
        label="$LEFT$",
    )
    axarr[3].plot(
        sim_f2t,
        simulation.phidot_r * speed_ratio,
        lw=6,
        color=orange_darker,
        label="$\phi_R$",
        zorder=-2,
    )
    axarr[3].plot(
        sim_f2t,
        simulation.phidot_l * speed_ratio,
        lw=6,
        color=blue_darker,
        label="$\phi_L$",
        zorder=-2,
    )
    axarr[3].legend()
    axarr[3].set(xlabel="time (s)", ylabel="speed (cm/s)$", xlim=tlim)

    clean_axes(f)

    # save figure
    save_figure(
        f,
        paths.folder_2WDD
        / "ANALYSIS"
        / f"sim2real_trial_{trialn}_sos_{sos_th}.png",
        close=True,
    )
    # break


# %%
