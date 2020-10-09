# %%
from pathlib import Path
from rich.progress import track
import numpy as np
import pyinspect as pi
import matplotlib.pyplot as plt
import shutil

from fcutils.maths.geometry import calc_distance_between_points_in_a_vector_2d

from proj.paths import analysis_fld, db_app
from proj.utils.misc import load_results_from_folder
from proj.environment.trajectories import compute_trajectory_stats

from analysis.utils import check_two_conf_equal

# %%
# Load all data in DB upload folder
flds = [f for f in Path(db_app).glob("*") if f.is_dir()]


loaded = dict(trajectory=[], history=[], cost_history=[],)

config = None

f, ax = plt.subplots()

sim_lengths = []
for fld in track(flds):
    _config, trajectory, history, cost_history = load_results_from_folder(fld)

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
"""

traj_dist = []
actual_dist = []

for traj, hist in track(
    zip(loaded["trajectory"], loaded["history"]),
    total=len(loaded["trajectory"]),
):
    metad = compute_trajectory_stats(
        traj, 1, config["trajectory"], config["planning"], mute=True
    )[-1]
    traj_dist.append(metad["distance_travelled"])

    actual_dist.append(
        np.sum(calc_distance_between_points_in_a_vector_2d(hist.x, hist.y))
        + config["trajectory"]["dist_th"]
    )

# %%
f, ax = plt.subplots(figsize=(12, 12))
ax.scatter(
    traj_dist,
    actual_dist,
    s=100,
    color="g",
    alpha=0.4,
    lw=1,
    ec=[0.3, 0.3, 0.3],
)
ax.plot([400, 1000], [400, 1000], lw=2, color=[0.6, 0.6, 0.6], zorder=-1)

ax.set(xlabel="Trajectory length", ylabel="Distance travelled")
# %%
