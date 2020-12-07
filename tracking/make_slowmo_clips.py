"""
    Script to generate slow motion clips of mice running in an
    open arena while filmed from below (Zane's data)
"""
from pathlib import Path
import numpy as np
import os

from fcutils.maths.utils import derivative, rolling_mean

from tracking._tracking import prepare_tracking_data

fps = 60


def run():
    # Get file paths
    folder = Path("Z:\\swc\\branco\\Zane\\experiments\\batch2_session1")
    videos = [f for f in folder.glob("*escconcat.avi")]
    h5s = [f for f in folder.glob("*.h5") if "escconcat" in f.name]

    dest_folder = Path(
        "D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\control\\behav_data\\Zane"
    )

    # Loop over pose files
    for posefile in h5s:
        # Get video corresponding to this file
        name = posefile.name.split("DeepCut")[0]
        video = [v for v in videos if name in v.name][0]
        print(f"Processing: {video.name}")

        # Get tracking data
        tracking = prepare_tracking_data(
            str(posefile),
            median_filter=True,
            interpolate_nans=True,
            likelihood_th=0.9,
        )

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
            if np.any(s[start:end] < 5):
                continue
            if end - start < 20:
                continue

            # use ffmpeg to create a new video
            out_vid = (
                dest_folder / f'{video.name.split(".")[0]}_{run_number}.mp4'
            )

            # start_time = time.strftime('%H:%M:%S', start * fps)
            # duration = time.strftime('%H:%M:%S', (end - start) * fps)
            # command = f'ffmpeg -i "{video}" -ss {start}  -t {end-start} "{out_vid}" -y'
            command = f'ffmpeg -ss {start} -i "{video}"  -t {end-start} "{out_vid}" -y -c:v copy -vcodec libx264 -crf 28'

            os.system(command)


if __name__ == "__main__":
    import sys

    sys.path.append("./")

    run()
