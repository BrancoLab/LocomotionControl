# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")
sys.path.append("/Users/federicoclaudi/Documents/Github/LocomotionControl")

import matplotlib.pyplot as plt
import numpy as np

from myterial import blue_darker, pink_darker

from analysis.load import load_complete_bouts
from data.data_utils import register_in_time, mean_and_std
from kinematics import track
import draw

"""
    Plots complete bouts through the arena, 
    but with the linearized track also
"""

# get linearized track
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


# load and clean complete bouts
bouts = load_complete_bouts(
    keep=100, duration_max=6, linearize_to=center_line, window=10
)

# TODO check if length of linearized matches that of actual bout

# %%
f = plt.figure(figsize=(16, 12))
axes = f.subplot_mosaic(
    """
        AAADDDIII
        AAAEEELLL
        AAAFFFMMM
        BBBGGGNNN
        CCCHHHOOO
    """
)

_ = draw.Hairpin(ax=axes["A"])
for n, bout in enumerate(bouts):
    # draw 2d and linearized tracking
    # draw.Tracking.scatter(bout.x, bout.y, c=bout.linearized.x, ax=axes["A"], vmin=0, vmax=260, cmap='tab20')
    # draw.Tracking.scatter(bout.linearized.x, bout.linearized.y, c=bout.linearized.x, ax=axes["B"], s=25, alpha=.5, vmin=0, vmax=260, cmap='tab20')
    draw.Tracking(bout.x, bout.y, ax=axes["A"])
    draw.Tracking(bout.linearized.x, bout.linearized.y, ax=axes["B"])

    # draw speed accel and ang vel along track
    axes["D"].plot(bout.linearized.x, bout.speed, color="k", alpha=0.5)
    axes["E"].plot(
        bout.linearized.x, bout.acceleration_mag, color="k", alpha=0.5
    )
    axes["F"].plot(bout.linearized.x, bout.thetadot, color="k", alpha=0.5)
    axes["G"].plot(bout.linearized.x, bout.thetadotdot, color="k", alpha=0.5)

    # draw speed accel and ang vel over time
    axes["I"].plot(bout.speed, color="k", alpha=0.5)
    axes["L"].plot(bout.acceleration_mag, color="k", alpha=0.5)
    axes["M"].plot(bout.thetadot, color="k", alpha=0.5)
    axes["N"].plot(bout.thetadotdot, color="k", alpha=0.5)

    # draw distance along the track for each path
    # axes["O"].scatter(bout.frames, bout.linearized.x, c=bout.linearized.x, s=25, alpha=.5, vmin=0, vmax=260, cmap='tab20')
    axes["O"].plot(bout.frames, bout.linearized.x, color="k")

draw.Hist([bout.duration for bout in bouts], ax=axes["H"])

# draw track @d and linearized
draw.draw_track(center_line, left_line, right_line, ax=axes["A"])
draw.draw_track(center_to_track, left_to_track, right_to_track, ax=axes["B"])

# plot curvature vs mean speed
K = center_line.curvature / center_line.curvature.max()
# axes["C"].scatter(center_line.comulative_distance, K, c=center_line.comulative_distance, s=25, alpha=.5, vmin=0, vmax=260, cmap='tab20')
axes["C"].plot(
    center_line.comulative_distance, K, color="k", label="norm. curvature"
)

mean_speed = register_in_time([bout.speed for bout in bouts])
M = mean_and_std(mean_speed)[0]

mean_avel = register_in_time([bout.thetadot for bout in bouts])
T = mean_and_std(mean_avel)[0]

mean_x = register_in_time([bout.linearized.x for bout in bouts])
X = mean_and_std(mean_x)[0]
axes["C"].plot(X, M / M.max(), lw=2, color=blue_darker, label="norm. speed")
axes["C"].plot(
    X, np.abs(T / T.max()), lw=2, color=pink_darker, label=r"$|\dot \theta|$"
)

for ax in "EFGLMN":
    axes[ax].axhline(0, lw=2, color="k", zorder=-1, alpha=0.5)

axes["C"].legend()
axes["B"].set(title="linearized track")
axes["C"].set(title="linearized track curvature")
axes["D"].set(title="speed", xlabel="track position")
axes["E"].set(title="acceleration", xlabel="track position")
axes["F"].set(
    title="angular velocity", ylim=[-800, 800], xlabel="track position"
)
axes["G"].set(
    title="angular acceleration", ylim=[-100, 100], xlabel="track position"
)
axes["I"].set(title="speed", xlabel="time (frames)")
axes["L"].set(title="acceleration", xlabel="time (frames)")
axes["M"].set(
    title="angular velocity", ylim=[-800, 800], xlabel="time (frames)"
)
axes["N"].set(
    title="angular acceleration", ylim=[-100, 100], xlabel="time (frames)"
)
_ = axes["O"].set(
    title="track position", xlabel="time (frames)", ylabel="track distance"
)

_ = f.tight_layout()


# %%
plt.show()
