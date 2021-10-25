# %%
import sys

sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from data.dbase.db_tables import ROICrossing

from data import paths
from data.data_utils import convolve_with_gaussian, register_in_time
from draw import colors
from analysis.fixtures import PAWS

import draw
from geometry import Path

# %%
ROI = 'T3'
bouts = pd.read_hdf(
        paths.analysis_folder
        / "behavior"
        / "roi_crossings"
        / f"{ROI}_crossings.h5"
    ).sort_values('duration').iloc[100:200]

# %%

f = plt.figure(figsize=(20, 10))
axes = f.subplot_mosaic(
    """
        AABBDD
        AACCEE
    """
)

draw.ROI(ROI, ax=axes['A'], 
        set_ax=True,
        img_path=r'C:\Users\Federico\Documents\GitHub\pysical_locomotion\draw\hairpin.png')
for n, (i, cross) in enumerate(bouts.iterrows()):
    start = np.where(cross.acceleration > 1)[0][0]
    # if n != 40:
    #     continue
    # _path = Path(
    #     convolve_with_gaussian(cross.x, kernel_width=11),
    #     convolve_with_gaussian(cross.y, kernel_width=11)
    #     )
    # _path = Path(
    #     medfilt(cross.x, kernel_size=5),
    #     medfilt(cross.y, kernel_size=5)
    #     )
    # create a path for the velocity and one for the accelertaion computation
    path = Path(
        cross.x, 
        cross.y,
        )

    # accel_path = Path(
    #     convolve_with_gaussian(path.velocity.x, kernel_width=11), 
    #     convolve_with_gaussian(path.velocity.y, kernel_width=11)
    # )
    time = np.linspace(0, 1, len(path.x))


    draw.Tracking.scatter(path.x, path.y,c=cross.gcoord,  ax=axes['A'])

    # axes['B'].plot(time, path.speed, color='k', alpha=.25)
    axes['B'].plot(cross.gcoord, convolve_with_gaussian(path.speed, kernel_width=3), color='k', alpha=.25)


    # plot acceleration
    # acceleration = path.acceleration.dot(path.tangent)/ path.tangent.magnitude
    acceleration = path.acceleration.dot(path.tangent)/ path.tangent.magnitude



    # axes['C'].plot(cross.gcoord, acceleration, color='r', alpha=.2)
    axes['C'].plot(cross.gcoord, convolve_with_gaussian(acceleration, kernel_width=3), color='r', alpha=.2)

    at_zero = np.where(acceleration < 0)[0][0]
    # axes['C'].scatter(time[at_zero], acceleration[at_zero], 
    #             color='k', alpha=1, zorder=100)

    draw.Tracking.scatter(path.x[at_zero], path.y[at_zero], color='k', zorder=100, ax=axes['A'])

    # plot avel and ang accel
    # axes['D'].plot(path.theta) 

    # axes['D'].plot(_path.acceleration.magnitude)
    # axes['E'].plot(accel_path.velocity.magnitude)

axes['C'].axhline(0)

# %%
f = plt.figure(figsize=(20, 10))
axes = f.subplot_mosaic(
    """
        AABBDD
        AACCEE
    """
)


for i, cross in bouts.iterrows():
    # path = Path(cross.x, cross.y)
    Path(
        convolve_with_gaussian(cross.x, kernel_width=11),
        convolve_with_gaussian(cross.y, kernel_width=11)
        )
    # draw.Tracking(path.x, path.y, ax=axes['A'])

    for G in np.arange(0.70, 0.90, 0.025):
        at_G = np.where(cross.gcoord >= G)[0][0]
        # draw.Arrow(path.x[at_G], path.y[at_G], path.velocity.angle[at_G], ax=axes['A'], color='salmon', outline=True, alpha=.5)
        # draw.Arrow(path.x[at_G]+2, path.y[at_G], path.acceleration.angle[at_G], ax=axes['A'], color='blue', outline=True, alpha=.5)
        draw.Arrow(0, 0, path.velocity.angle[at_G], ax=axes['A'], color='salmon', outline=True, alpha=.5)
        draw.Arrow(2, 0, path.acceleration.angle[at_G], ax=axes['A'], color='blue', outline=True, alpha=.5)
        draw.Arrow(4, 0, path.acceleration.angle[at_G] - path.acceleration.angle[at_G+1], ax=axes['A'], color='green', outline=True, alpha=.5)

        break
    # break
