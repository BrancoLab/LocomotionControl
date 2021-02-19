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
    blue_darker,
    orange_darker,
    blue_light,
    orange_light,
    blue_grey,
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
from control.utils import get_theta_from_xy

# %%
# Get simulations and lookup
simulations_folder = Path(
    "Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD\\SIMULATIONS"
)

simulations_lookup2 = {}
for sim in path.subdirs(simulations_folder):
    try:
        trial = pd.read_hdf(sim / "trial.h5")
    except FileNotFoundError:
        logger.warning(f"Did not find trial file for {sim}")
        continue
    simulations_lookup2[trial["name"]] = sim
logger.info(f"Loaded simulations for {len(simulations_lookup2)} trials")

trials = Trials(only_tracked=True)

BAD_TRIALS = ()  # trials in which the simulation went wrong

# %%
"""
    For each trial, plot the real trial DATA and the results from the simulation
"""
sos_th = 1
sos = butter(6, sos_th, output="sos", fs=60)

for trialn in track(range(len(trials)), total=len(trials)):
    if trialn in BAD_TRIALS:
        continue

    # load all data
    trial = trials[trialn]
    if not trial.good or len(trial) < 50:
        continue

    try:
        simulated = pd.read_hdf(
            simulations_lookup2[trial.name] / "trial.h5"
        )  # trial for simulation
    except KeyError:
        logger.warning(
            f"Skipping {trial.name} [{trialn}] as no subfolder was found"
        )
        continue

    simulation = pd.read_hdf(
        simulations_lookup2[trial.name] / "history.h5"
    )  # simulation results
    simulation["dirmvmt"] = get_theta_from_xy(simulation.x, simulation.y)
    # simulation.dirmvmt[0] = simulation.dirmvmt[1].copy()
    # simulation.dirmvmt = simulation.dirmvmt

    assert trial.name == simulated["name"], "loaded the wrong one mate"

    # get sim start wrt to trial start
    try:
        sim_start_idx = find_nearest(trial.body.y, simulation.y[0])
    except IndexError:
        continue

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
    f, axarr = plt.subplots(figsize=(16, 9), nrows=2, ncols=3)
    axarr = axarr.flatten()

    # plot XY
    axarr[0].plot(
        simulated.x,
        simulated.y,
        lw=8,
        color=blue_grey,
        label="trajectory",
        zorder=-1,
    )
    plot_line_outlined(
        axarr[0],
        simulation.x[::20],
        simulation.y[::20],
        color="salmon",
        lw=8,
        outline=2,
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
        zorder=4,
    )

    axarr[0].axis("equal")
    axarr[0].set(xlabel="X (cm)", ylabel="Y (cm)")
    axarr[0].legend()

    # plot speed
    axarr[3].plot(
        trial_f2t,
        rolling_mean(trial.v, 12),
        lw=8,
        color=blue_grey,
        label="real",
    )
    plot_line_outlined(
        axarr[3],
        sim_f2t,
        simulation.v,
        lw=8,
        color="salmon",
        label="simulation",
    )
    axarr[3].set(xlabel="time (s)", ylabel="$v$", xlim=tlim)
    axarr[3].legend()

    # plot paw speeds raw
    axarr[2].plot(
        trial_f2t,
        rolling_mean(trial.right_hl.speed, 5),
        lw=8,
        zorder=-1,
        color=orange_light,
        label="$right hind limb$",
    )
    axarr[1].plot(
        trial_f2t,
        rolling_mean(trial.left_hl.speed, 5),
        lw=8,
        zorder=-1,
        color=blue_light,
        label="$left hind limb$",
    )
    axarr[2].plot(
        trial_f2t,
        rolling_mean(trial.right_fl.speed, 5),
        lw=2,
        ls="--",
        zorder=-1,
        color=orange_light,
    )
    axarr[1].plot(
        trial_f2t,
        rolling_mean(trial.left_fl.speed, 5),
        lw=2,
        ls="--",
        zorder=-1,
        color=blue_light,
    )
    plot_line_outlined(
        axarr[2],
        sim_f2t,
        simulation.phidot_r * speed_ratio,
        lw=6,
        color=orange_darker,
        label="right wheel",
        outline=2,
    )
    plot_line_outlined(
        axarr[1],
        sim_f2t,
        simulation.phidot_l * speed_ratio,
        lw=6,
        color=blue_darker,
        label="left wheel",
        outline=2,
    )
    axarr[2].legend()
    axarr[2].set(xlabel="time (s)", ylabel="speed (cm/s)$", xlim=tlim)
    axarr[1].legend()
    axarr[1].set(xlabel="time (s)", ylabel="speed (cm/s)$", xlim=tlim)

    # plot paw speeds processed
    axarr[5].plot(
        trial_f2t,
        rolling_mean((trial.right_hl.speed + trial.right_fl.speed) / 2, 20),
        lw=8,
        zorder=-1,
        color=orange_light,
        label="$RIGHT$",
    )
    axarr[4].plot(
        trial_f2t,
        rolling_mean((trial.left_hl.speed + trial.left_fl.speed) / 2, 20),
        lw=8,
        zorder=-1,
        color=blue_light,
        label="$LEFT$",
    )
    plot_line_outlined(
        axarr[5],
        sim_f2t,
        rolling_mean(simulation.phidot_r * speed_ratio, 20),
        lw=6,
        color=orange_darker,
        label="right wheel",
        outline=2,
    )
    plot_line_outlined(
        axarr[4],
        sim_f2t,
        rolling_mean(simulation.phidot_l * speed_ratio, 20),
        lw=6,
        color=blue_darker,
        label="left wheel",
        outline=2,
    )
    axarr[4].legend()
    axarr[4].set(xlabel="time (s)", ylabel="speed (cm/s)$", xlim=tlim)
    axarr[5].legend()
    axarr[5].set(xlabel="time (s)", ylabel="speed (cm/s)$", xlim=tlim)

    clean_axes(f)

    # plt.show()
    # break

    # save figure
    save_figure(
        f,
        Path(
            r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\experimental_validation\2WDD\analysis\trials"
        )
        / f"sim2real_trial_{trial['name']}.png",
        close=True,
    )
    # break


# %%
