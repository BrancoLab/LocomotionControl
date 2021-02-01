"""
    Script to generate slow motion clips of mice running in an
    open arena while filmed from below (Zane's data)
"""
from pathlib import Path
import numpy as np
import pandas as pd

from fcutils.maths.signals import derivative, rolling_mean
from fcutils.video import trim_clip

from tracking._tracking import (
    prepare_tracking_data,
    compute_body_segments,
    average_body_angle,
    wrapdiff,
)

fps = 60

bones = dict(
    upper_body=("snout", "body"),
    lower_body=("body", "tail_base"),
    left_diag=("left_forepaw", "right_hindpaw"),
    right_diag=("right_forepaw", "left_hindpaw"),
    left=("left_forepaw", "left_hindpaw"),
    right=("right_forepaw", "right_hindpaw"),
)


def collate_tracking(tracking, bones_tracking):
    """ put all tracking into a single dataframe """
    data = {}
    for bp, tr in tracking.items():
        data.update({bp + "_" + k: tr[k].values for k in tr.columns})

    for bone, tr in bones_tracking.items():
        data.update({bone + "_" + k: tr[k].values for k in tr.columns})

    return pd.DataFrame(data)


def run():
    # Get file paths
    folder = Path(
        r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\control\behav_data\ZaneRaw"
    )

    videos = [f for f in folder.glob("*escconcat.avi")]
    h5s = [f for f in folder.glob("*.h5") if "escconcat" in f.name]

    dest_folder = Path(
        r"D:\Dropbox (UCL)\Rotation_vte\Locomotion\control\behav_data\ZaneClips"
    )

    # Loop over pose files
    for posefile in h5s:
        # Get video corresponding to this file
        name = posefile.name.split("DeepCut")[0]
        video = [v for v in videos if name in v.name][0]
        print(f"Processing: {posefile.name} -> {video.name}")

        # Get tracking data
        tracking = prepare_tracking_data(
            str(posefile),
            median_filter=True,
            interpolate_nans=True,
            likelihood_th=0.8,
        )

        bones_tracking = compute_body_segments(tracking, bones)
        body_angle = average_body_angle(
            bones_tracking["upper_body"].bone_orientation.values,
            bones_tracking["lower_body"].bone_orientation.values,
        )

        # collated = collate_tracking(tracking, bones_tracking)

        y, s = (
            tracking["body"].y.values,
            tracking["body"].speed.values,
        )
        s = rolling_mean(s, 10)

        # Get start/end of clean runs
        run = np.zeros_like(y)
        run[(y > 500) & (y < 1500)] = 1
        starts = np.where(derivative(run) > 0)[0]
        ends = np.where(derivative(run) < 0)[0]
        ends = [e for e in ends if e > starts[0]]

        # Iterate runs
        for run_number, (start, end) in enumerate(zip(starts, ends)):
            if np.any(s[start:end] < 5):  # to slow
                continue
            if end - start < 35:  # to short
                continue

            # Check turn angle
            turn = wrapdiff(np.array(body_angle[end] - body_angle[start]))
            print(int(turn))
            if np.abs(turn) < 40:
                continue

            # take a bit before and after
            # start -= int(0.2 * fps)
            # end += int(0.2 * fps)

            # create a new video
            out_vid = (
                dest_folder
                / f'turn_{int(np.abs(turn))}_{video.name.split(".")[0]}_{run_number}.mp4'
            )

            trim_clip(
                str(video),
                str(out_vid),
                frame_mode=True,
                start_frame=start,
                stop_frame=end,
                sel_fps=4,
            )

            # Save tracking data
            # out_tracking = (
            #     dest_folder / f'{video.name.split(".")[0]}_{run_number}.h5'
            # )
            # collated[start:end].to_hdf(out_tracking, key="hdf")


if __name__ == "__main__":
    import sys

    sys.path.append("./")

    run()
