import h5py
import numpy as np
import pandas as pd

import sys

sys.path.append("./")
from tracking._tracking import prepare_tracking_data, compute_body_segments

# --------------------------- load from matlab file --------------------------- #

# load tracking for all markers
f = "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/control/behav_data/capture/nolj_Recording_day8_caff1_nolj_imputed.mat"
f = h5py.File(f, "r")

# only use some markers
markers = list(f["markers_preproc"].keys())
main_markers = ["ArmL", "ArmR", "ShinL", "ShinR", "SpineM", "SpineF", "SpineL"]

# keep only XY position of each marker and make it look like DLC tracking
data = {m: np.array(f["markers_preproc"][m]) for m in main_markers}
data = {
    m: pd.DataFrame(
        {"x": d[0, :], "y": d[1, :], "likelihood": np.ones_like(d[0, :])}
    )
    for m, d in data.items()
}

# --------------------------------- clean up --------------------------------- #

# clean up
print("Processing body parts")
tracking = prepare_tracking_data(
    tracking=data,
    bodyparts=data.keys(),
    median_filter=True,
    filter_kwargs={"kernel": 61},
    smooth_dir_mvmt=False,
    verbose=True,
)

# skeleton
print("Processing skeleton")
skeleton = dict(
    body1=("SpineM", "SpineF"),
    body2=("SpineM", "SpineL"),
    body3=("SpineF", "SpineL"),
)

bones = compute_body_segments(tracking, skeleton)

# ------------------------------- put together ------------------------------- #
tracking_vars = (
    "x",
    "y",
    "speed",
    "direction_of_movement",
    "angular_velocity",
)
tracking_data = {
    f"{bp}_{var}": tracking[bp][var]
    for var in tracking_vars
    for bp in tracking.keys()
}

bone_vars = ("bone_length", "bone_orientation")
bone_data = {
    f"{bone}_{var}": bones[bone][var]
    for var in bone_vars
    for bone in bones.keys()
}

data = {**tracking_data, **bone_data}

print("saving")
pd.DataFrame(data).to_hdf(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/control/behav_data/capture_cleaned.h5",
    key="hdf",
)
