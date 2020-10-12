# %%
from pathlib import Path
from rich.progress import track
import numpy as np
import pyinspect as pi
import matplotlib.pyplot as plt
import shutil

from fcutils.maths.geometry import calc_distance_between_points_in_a_vector_2d

from proj.paths import analysis_fld, db_app
from proj.utils.misc import load_results_from_folder, duration_from_history
from proj.environment.trajectories import compute_trajectory_stats

from analysis.utils import check_two_conf_equal

# %%
# Load all data in DB upload folder
flds = [f for f in Path(db_app).glob("*") if f.is_dir()]


loaded = dict(trajectory=[], history=[], cost_history=[], trials=[])

config = None

f, ax = plt.subplots()

sim_lengths = []
for fld in track(flds):
    (
        _config,
        trajectory,
        history,
        cost_history,
        trial,
        info,
    ) = load_results_from_folder(fld)

    # Check that all data have the same config
    if config is None:
        config = _config
    else:
        if not check_two_conf_equal(config, _config):
            raise ValueError("Not all simulations have the same config")

    # stash
    loaded["trajectory"].append(trajectory)
    loaded["history"].append(history)
    loaded["cost_history"].append(cost_history)
    loaded["trials"].append(trial)

    sim_lengths.append(len(history))

plt.hist(sim_lengths)
pi.ok("Data loaded")

# %%
# Copy all outcome plots in same folder
outcomes_fld = Path(analysis_fld) / "outcomes"
outcomes_fld.mkdir(exist_ok=True)

for fld in track(flds):
    src = list(fld.glob("outcome.png"))[0]
    dst = outcomes_fld / (src.parent.name + "_outcome.png")

    shutil.copy(src, dst)

pi.ok("All outcome images copied", str(outcomes_fld))

# %%
"""
    Look at trajectory length vs distance travelled
    and escape duration vs trajectory duration
"""

traj_dist, traj_dur = [], []
actual_dist, actual_dur = [], []

for traj, hist in track(
    zip(loaded["trajectory"], loaded["history"]),
    total=len(loaded["trajectory"]),
):
    last_idx = hist.trajectory_idx.values[-1]

    # TODO get trial FPS and duration from that
    duration = len(traj[:last_idx, :]) / 40  # get FPS

    metad = compute_trajectory_stats(
        traj[:last_idx, :],
        duration,
        config["trajectory"],
        config["planning"],
        mute=True,
    )[-1]

    traj_dist.append(metad["distance_travelled"])
    traj_dur.append(metad["duration"])

    actual_dist.append(
        np.sum(calc_distance_between_points_in_a_vector_2d(hist.x, hist.y))
    )
    actual_dur.append(duration_from_history(hist, config))

f, axarr = plt.subplots(ncols=2, figsize=(18, 9))

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
axarr[0].plot([25, 130], [25, 130], lw=2, color=[0.6, 0.6, 0.6], zorder=-1)

axarr[0].set(xlabel="Trajectory length", ylabel="Distance travelled")

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
# axarr[1].plot([25, 130], [25, 130], lw=2, color=[0.6, 0.6, 0.6], zorder=-1)

axarr[1].set(xlabel="Trajectory duration", ylabel="Simulation duration")

# %%
"""
# TODO look 
    - distance from XY trajectory
    - distance from speed trajectory
    - total ammount of turn vs expected
    - duration vs expected

"""
