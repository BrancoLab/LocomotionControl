import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from seaborn import regplot
import numpy as np
from fcutils.plot.figure import clean_axes

from myterial import blue_grey_darker, salmon, indigo

import sys

sys.path.append("./")

from tracking.gait import print_steps_summary

# --------------------------------- load data -------------------------------- #

# load all steps from each mouse
folder = Path(
    "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\control\\experimental_validation"
)

steps_files = [f for f in folder.glob("*.h5") if "steps" in f.name]

all_steps = []
for sfile in steps_files:
    all_steps.append(pd.read_hdf(sfile, key="hdf"))

steps = pd.concat(all_steps)

# ? select some steps
ANGLE_TH = 10  # any less and you're not turning
turn = steps.loc[np.abs(steps.angle_delta) > ANGLE_TH]
straight = steps.loc[np.abs(steps.angle_delta) <= ANGLE_TH]
print_steps_summary(turn)

# ----------------------------------- plot ----------------------------------- #

# make figure
f, ax = plt.subplots(figsize=(10, 10))
ax.axvline(0, ls="--", color=[0.5, 0.5, 0.5])
ax.axhline(0, ls="--", color=[0.5, 0.5, 0.5])

x, y = 5, 45  # axes lims
ax.set(
    title="stride difference vs turn angle",
    xlim=[-x, x],
    xticks=[-x, 0, x],
    xticklabels=[f"R>L\n{-x}", "R=L\n0", f"R<L\n{x}"],
    xlabel="(L-R) stride-delta\m(cm)",
    ylim=[-y, y],
    yticks=[-y, 0, y],
    yticklabels=[f"turn\nleft\n{-y}", "no\nturn\n0", f"turn\nright\n{y}"],
    ylabel="(end-start) angle-delta\n(deg)",
)

colors = [salmon if s.side == "L" else indigo for i, s in turn.iterrows()]

#  plot all steps
ax.scatter(
    turn.stride_delta,
    turn.angle_delta,
    c=colors,
    s=45,
    lw=0.5,
    edgecolors=[0.2, 0.2, 0.2],
)

ax.scatter(
    straight.stride_delta, straight.angle_delta, color=[0.6, 0.6, 0.6], s=45,
)
# ------------------------------ lin. regression ----------------------------- #

for data, color in zip((turn, straight), ("magenta", blue_grey_darker)):
    regplot(
        "stride_delta",
        "angle_delta",
        data,
        scatter=False,
        truncate=True,
        robust=True,
        line_kws={"color": color},
    )


clean_axes(f)
plt.show()
