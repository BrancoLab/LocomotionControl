import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("./")

from myterial import blue, pink

import draw
from geometry import Path


"""
    Plot a few ROI crossings with vectors overlayed
"""

ROI = "T1"

# load tracking
bouts = pd.read_hdf(
    f"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/roi_crossings/{ROI}_crossings.h5"
)
selected_bouts = bouts.sample(6).reset_index()


# draw stuff
f, axes = plt.subplots(figsize=(18, 12), ncols=3, nrows=2)
axes = axes.flatten()

for i, bout in selected_bouts.iterrows():
    # generate a path from tracking
    path = Path(bout.x, bout.y)

    ax = axes[i]

    # draw arena and tracking
    draw.ROI(ROI, set_ax=True, shade=False, ax=ax)
    # draw.Tracking(bouts.x, bouts.y, alpha=0.5)

    # draw bout tracking and velocity/acceleration vectors
    # draw.Tracking.scatter(path.x, path.y, c=path.tangent.angle, cmap='bwr', vmin=-180, vmax=180)
    draw.Tracking(path.x, path.y, color=[0.6, 0.6, 0.6], lw=5, ax=ax)

    draw.Arrows(
        path.x,
        path.y,
        path.velocity.angle,
        L=path.velocity.magnitude,
        color=blue,
        step=2,
        outline=True,
        width=2,
        ax=ax,
    )
    draw.Arrows(
        path.x,
        path.y,
        path.acceleration.angle,
        L=path.acceleration.magnitude * 2,
        color=pink,
        step=2,
        outline=True,
        width=2,
        ax=ax,
    )

f.tight_layout()
plt.show()
