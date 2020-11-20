# %%
from pathlib import Path
from rich.progress import track
import numpy as np
import pyinspect as pi
import matplotlib.pyplot as plt
import shutil

from fcutils.maths.geometry import (
    calc_distance_between_points_in_a_vector_2d,
    calc_angle_between_points_of_vector_2d,
    calc_distance_between_points_two_vectors_2d,
)
from fcutils.plotting.utils import clean_axes
import sys

sys.path.append("../")
from proj.paths import analysis_fld, db_app
from proj.utils.misc import load_results_from_folder, duration_from_history
from proj.environment.trajectories import compute_trajectory_stats

# from analysis.utils import check_two_conf_equal

# %%
# Load all data in DB upload folder
flds = [f for f in Path(db_app).glob("*") if f.is_dir()]


loaded = dict(
    trajectory=[], history=[], cost_history=[], trials=[], duration=[]
)

config = None

f, ax = plt.subplots()

sim_lengths = []
for fld in track(flds):
    try:
        (
            _config,
            trajectory,
            history,
            cost_history,
            trial,
            info,
        ) = load_results_from_folder(fld)
    except ValueError:
        print("skipping")
        continue

    # Check that all data have the same config
    if config is None:
        config = _config
    # else:
    #     if not check_two_conf_equal(config, _config):
    #         raise ValueError("Not all simulations have the same config")

    # stash
    loaded["trajectory"].append(trajectory)
    loaded["history"].append(history)
    loaded["cost_history"].append(cost_history)
    loaded["trials"].append(trial)
    loaded["duration"].append(info["traj_duration"])

    sim_lengths.append(len(history))

plt.hist(sim_lengths)
pi.ok("Data loaded")

# %%
# Copy all outcome plots in same folder
outcomes_fld = Path(analysis_fld) / "outcomes"
outcomes_fld.mkdir(exist_ok=True)

for fld in track(flds):
    try:
        src = list(fld.glob("outcome.png"))[0]
    except Exception:
        continue
    dst = outcomes_fld / (src.parent.name + "_outcome.png")

    shutil.copy(src, dst)

pi.ok("All outcome images copied", str(outcomes_fld))

# %%
"""
    Look at trajectory length vs distance travelled
    and escape duration vs trajectory duration
"""

traj_ang, traj_dist, traj_dur = [], [], []
actual_ang, actual_dist, actual_dur = [], [], []


for traj, hist, dur in track(
    zip(loaded["trajectory"], loaded["history"], loaded["duration"]),
    total=len(loaded["trajectory"]),
):
    last_idx = hist.trajectory_idx.values[-1]

    # Get trajectory metadata
    # traj = traj[:last_idx, :]
    metad = compute_trajectory_stats(
        traj, 1, config["trajectory"], config["planning"], mute=True,
    )[-1]

    traj_dist.append(metad["distance_travelled"])
    traj_dur.append(dur)
    traj_ang.append(
        np.sum(calc_angle_between_points_of_vector_2d(traj[:, 0], traj[:, 1]))
        / len(traj)
    )

    # Compute stuff on simulation
    actual_dist.append(
        np.sum(calc_distance_between_points_in_a_vector_2d(hist.x, hist.y))
    )
    actual_dur.append(duration_from_history(hist, config))
    actual_ang.append(
        np.sum(calc_angle_between_points_of_vector_2d(hist.x, hist.y))
        / len(hist)
    )


# ? Plotting
f, axarr = plt.subplots(ncols=2, nrows=2, figsize=(18, 15))
axarr = axarr.flatten()

# Plot distance
axarr[0].scatter(
    traj_dist,
    actual_dist,
    s=100,
    color="g",
    alpha=0.4,
    lw=1,
    ec=[0.3, 0.3, 0.3],
)
axarr[0].plot([90, 200], [90, 200], lw=2, color=[0.6, 0.6, 0.6], zorder=-1)

axarr[0].set(
    xlabel="Trajectory length", ylabel="Distance travelled", title="Distance"
)

# Plot duration
axarr[1].scatter(
    traj_dur,
    actual_dur,
    s=100,
    color=[0.4, 0.4, 0.8],
    alpha=0.4,
    lw=1,
    ec=[0.3, 0.3, 0.3],
)
axarr[1].plot([0.5, 6], [0.5, 6], lw=2, color=[0.6, 0.6, 0.6], zorder=-1)
axarr[1].set(
    xlabel="Trajectory duration",
    ylabel="Simulation duration",
    title="Duration",
)

# Plot turn angle
# Plot duration
axarr[2].scatter(
    traj_ang,
    actual_ang,
    s=100,
    color=[0.8, 0.4, 0.4],
    alpha=0.4,
    lw=1,
    ec=[0.3, 0.3, 0.3],
)
axarr[2].plot([80, 320], [80, 320], lw=2, color=[0.6, 0.6, 0.6], zorder=-1)
axarr[2].set(
    xlabel="Trajectory norm angle",
    ylabel="Simulation norm angle",
    title="Turn angle",
)

for ax in axarr:
    ax.axis("equal")
axarr[-1].axis("off")
_ = clean_axes(f)


# %%
"""
    Compute normalized distance from trajectory
"""
xy_dist, v_dist = [], []

for traj, hist in track(
    zip(loaded["trajectory"], loaded["history"]),
    total=len(loaded["trajectory"]),
):
    last_idx = hist.trajectory_idx.values[-1]

    # Get trajectory point at each simulation step
    traj_sim = np.vstack([traj[i, :] for i in hist.trajectory_idx])

    # Get tracking
    sim = np.vstack([hist.x, hist.y])

    xy_dist.append(
        np.sum(
            calc_distance_between_points_two_vectors_2d(traj_sim[:, :2], sim.T)
        )
        / len(hist)
    )

    v_dist.append(np.sum(np.sqrt((traj_sim[:, 2] - hist.v) ** 2)) / len(hist))

f, ax = plt.subplots(figsize=(10, 10))

ax.hist(
    xy_dist,
    color=[0.5, 0.5, 0.5],
    bins=15,
    density=True,
    alpha=0.5,
    label="XY traj",
)
ax.hist(v_dist, color="m", bins=15, density=True, alpha=0.5, label="Speed")

ax.legend()
_ = ax.set(xlabel="Normalized distance", ylabel="density")
