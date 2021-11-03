# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from tpd import recorder

from data import paths
from data.data_structures import LocomotionBout
from kinematics import track

import draw

"""
    Plots complete bouts through the arena, looking at linearized track
    quantities (e.g. speed vs track position)
"""

folder = Path(
    r"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab"
)
recorder.start(
    base_folder=folder.parent, folder_name=folder.name, timestamp=False
)

# %%
(
    left_line,
    center_line,
    right_line,
    left_to_track,
    center_to_track,
    right_to_track,
    control_points,
) = track.extract_track_from_image(
    points_spacing=1, restrict_extremities=False, apply_extra_spacing=False,
)


# %%
# load and clean complete bouts
_bouts = pd.read_hdf(
    paths.analysis_folder / "behavior" / "saved_data" / f"complete_bouts.h5"
)
_bouts = _bouts.loc[(_bouts.start_roi == 0) & (_bouts.duration < 8)]
_bouts = _bouts.sort_values("duration")  # .iloc[:5]
print(f"Kept {len(_bouts)} bouts")

bouts = []
for i, bout in track(_bouts.iterrows()):
    bouts.append(LocomotionBout(bout, linearize_to=center_line))


# %%
f = plt.figure(figsize=(12, 12))
axes = f.subplot_mosaic(
    """
        AAADDD
        AAAEEE
        AAAFFF
        BBBGGG
        CCCHHI
    """
)
f.tight_layout()

_ = draw.Hairpin(ax=axes["A"])
for n, bout in enumerate(bouts):
    # draw 2d and linearized tracking
    draw.Tracking(bout.x, bout.y, ax=axes["A"])

    draw.Tracking(bout.linearized.x, bout.linearized.y, ax=axes["B"])

    # draw speed accel and ang vel along track
    axes["D"].plot(bout.linearized.x, bout.speed, color="k", alpha=0.5)
    axes["E"].plot(bout.linearized.x, bout.acceleration, color="k", alpha=0.5)

    axes["F"].plot(bout.linearized.x, bout.thetadot, color="k", alpha=0.5)
    axes["G"].plot(bout.linearized.x, bout.thetadotdot, color="k", alpha=0.5)

    # mark peak acceleration etc
    # peak_accel = np.where(bout.acceleration > np.percentile(bout.acceleration, 80))[0]
    # draw.Tracking.scatter(bout.x[peak_accel], bout.y[peak_accel],
    #     color=blue, zorder=100, ax=axes['A'], label='peak accel' if n == 0 else None)

    # neg_peak_accel = np.where(bout.acceleration < np.percentile(bout.acceleration, 20))[0]
    # draw.Tracking.scatter(bout.x[neg_peak_accel], bout.y[neg_peak_accel],
    #     color=blue_darker, zorder=100, ax=axes['A'], label='peak decel' if n == 0 else None)

    # peak_ang_vel = np.where(np.abs(bout.thetadot) > np.percentile(np.abs(bout.thetadot), 85))[0]
    # draw.Tracking.scatter(bout.x[peak_ang_vel], bout.y[peak_ang_vel],
    #     color=pink, zorder=100, ax=axes['A'], label='peak ang vel' if n == 0 else None)

    # peak_ang_accl = np.where(np.abs(bout.thetadotdot) > np.percentile(np.abs(bout.thetadotdot), 85))[0]
    # draw.Tracking.scatter(bout.x[peak_ang_accl], bout.y[peak_ang_accl],
    #     color=pink_darker, zorder=100, ax=axes['A'], label='peak ang acc' if n == 0 else None)

    # draw distance along the track for each path
    axes["H"].plot(bout.linearized.x, color="k", alpha=0.5)

    # scatter speed vs ang acc
    axes["I"].scatter(
        bout.speed[(15 < bout.linearized.x) & (bout.linearized.x < 250)],
        np.abs(
            bout.thetadot[(15 < bout.linearized.x) & (bout.linearized.x < 250)]
        ),
        color="k",
        alpha=0.2,
    )


draw.draw_track(center_line, left_line, right_line, ax=axes["A"])

draw.draw_track(center_to_track, left_to_track, right_to_track, ax=axes["B"])

axes["C"].plot(center_line.comulative_distance, center_line.curvature)

for ax in "EFG":
    axes[ax].axhline(0, lw=2, color="k", zorder=-1)

# axes['A'].legend()
axes["B"].set(title="linearized track")
axes["C"].set(title="linearized track curvature")
axes["D"].set(title="speed")
axes["E"].set(title="acceleration")
axes["F"].set(title="angular velocity", ylim=[-800, 800])
axes["G"].set(title="angular acceleration")
_ = axes["H"].set(
    title="track position", xlabel="frames", ylabel="track distance"
)
_ = axes["I"].set(xlabel="speed", ylabel="ang vel", ylim=[0, 1000])

# %%
