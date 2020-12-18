from rich.progress import track
from pathlib import Path
import pandas as pd
import sys

sys.path.append("./")
from tracking._tracking import prepare_tracking_data, compute_body_segments


# get files
folder = Path(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/control/behav_data/Zane"
)

files = [f for f in folder.glob("*.h5") if "DLC" in f.name]

# define skeleton
skeleton = dict(
    head_l=("snout", "l_ear"),
    head_r=("snout", "r_ear"),
    body_whole=("snout", "tail_base"),
    body_upper=("snout", "body"),
    body_lower=("body", "tail_base"),
    left=("LF", "LH"),
    right=("RF", "RH"),
    right_diagonal=("RF", "LH"),
    left_diagonal=("LF", "RH"),
)


for f in track(files):
    # process data
    tracking = prepare_tracking_data(
        str(f),
        likelihood_th=0.999,
        median_filter=True,
        filter_kwargs={"kernel": 11},
        compute=True,
        smooth_dir_mvmt=True,
        interpolate_nans=True,
        verbose=False,
    )

    bones = compute_body_segments(tracking, skeleton)

    # put everything together:
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

    # save
    name = f.name.split("DLC")[0] + ".h5"
    pd.DataFrame(data).to_hdf(f.parent / name, key="hdf")

# %%
