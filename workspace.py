# %%


from behaviour.tracking.tracking import prepare_tracking_data
from fcutils.maths.geometry import (
    calc_angle_between_vectors_of_points_2d as get_bone_angle,
)
import numpy as np
import pandas as pd

# %%
path = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/behav_data/Zane/ZM_200913_ZM006_videoDeepCut_resnet50_DLCSep6shuffle1_200000.h5"


tracking = prepare_tracking_data(
    path, smooth_dir_mvmt=False, likelihood_th=0.8, interpolate_nans=True
)


body_orientation = get_bone_angle(
    tracking["body"].x.values,
    tracking["body"].y.values,
    tracking["tail_base"].x.values,
    tracking["tail_base"].y.values,
)


# %%
escapes_file = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/behav_data/Zane/escapes.txt"
with open(escapes_file, "r") as f:
    escapes = [int(e) for e in f.readlines()]

# %%
trials = {}
for bp in tracking.keys():
    trials[bp + "_xy"] = []
    trials[bp + "_speed"] = []

    if bp == "body":
        trials[bp + "_orientation"] = []


for escape in escapes:
    for bp in tracking.keys():
        start = escape
        end = start + 1000
        x = tracking[bp].x[start:end]
        y = tracking[bp].y[start:end]
        s = tracking[bp].speed[start:end]

        trials[bp + "_xy"].append(np.vstack([x, y]).T)
        trials[bp + "_speed"].append(s)

        if bp == "body":
            trials["body_orientation"].append(body_orientation[start:end])


trials = pd.DataFrame(trials)

# %%
import matplotlib.pyplot as plt

f, ax = plt.subplots()

for i, trial in trials.iterrows():
    angle = np.unwrap(np.radians(90 - trial.body_orientation))
    ax.scatter(trial.body_xy[:, 0], trial.body_xy[:, 1], c=angle)

ax.axis("equal")

# %%
f, ax = plt.subplots()
for bp in tracking.keys():
    ax.plot(tracking[bp].x[5000:5500], tracking[bp].y[5000:5500])

# %%
trials.to_hdf(
    "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/behav_data/zanes.h5",
    key="hdf",
)


# %%
