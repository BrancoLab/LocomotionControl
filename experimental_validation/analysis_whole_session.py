from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from fcutils.plotting.utils import clean_axes
from pyinspect.utils import dir_files
from myterial import (
    salmon_darker,
    indigo_darker,
    indigo,
    salmon,
    blue_grey_darker,
    orange,
)

folder = Path(
    "Z:\\swc\\branco\\Federico\\Locomotion\\control\\experimental_validation\\2WDD"
)
tracking_folder = folder / "TRACKING_DATA"


# plot all tracking from all sessions

f, ax = plt.subplots(figsize=(8, 14))

for path in dir_files(tracking_folder, "*_tracking.h5"):
    tracking = pd.read_hdf(path, key="hdf")

    x = tracking.body_x.values[::5]
    y = tracking.body_y.values[::5]
    s = tracking.body_speed.values[::5]
    angle = tracking.body_lower_bone_orientation.values[::5]

    ax.scatter(
        x,
        y,
        lw=0.75,
        c=angle,
        s=s,
        edgecolors=blue_grey_darker,
        alpha=0.8,
        cmap="bwr",
    )

ax.set(title="All tracking")
plt.show()
