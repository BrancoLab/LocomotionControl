import sys
import pandas as pd
import matplotlib.pyplot as plt
from celluloid import Camera
import pathlib
import numpy as np

sys.path.append("./")

from myterial import blue, pink
from fcutils.progress import track

from data import colors
import draw
from geometry import Path, GrowingPath
from geometry.interpolate import interpolate_at_frame

"""
    Creates an animation of a single ROI crossing showin the position of the mouse and 
    velocity/acceleration vectors
"""


def animate_roi_crossing(ROI: str, crossing_id: int, FPS: int):
    # ---------------------------------- params ---------------------------------- #
    ROI = "T1"
    crossing_id = 1
    save_folder = pathlib.Path(
        "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/roi_crossings_animations"
    )
    FPS = 10

    # ----------------------------------- prep ----------------------------------- #
    # load tracking
    bouts = pd.read_hdf(
        f"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/roi_crossings/{ROI}_crossings.h5"
    )
    bout = bouts.sort_values("duration").iloc[crossing_id]

    # generate a path from tracking
    path = Path(bout.x, bout.y)

    # create fig and start camera
    f = plt.figure(figsize=(8, 14))
    axes = f.subplot_mosaic(
        """
            AA
            AA
            BB
        """
    )
    camera = Camera(f)

    # ----------------------------------- plot ----------------------------------- #
    n_interpolated_frames = int(60 / FPS)
    n_frames = len(bout.x) - 1
    trajectory = GrowingPath()
    for frame in track(range(n_frames), total=n_frames):
        # repeat each frame N times interpolating between frames
        for interpol in np.linspace(0, 1, n_interpolated_frames):
            # get interpolated quantities
            x = interpolate_at_frame(path.x, frame, interpol)
            y = interpolate_at_frame(path.y, frame, interpol)
            speed = interpolate_at_frame(path.speed, frame, interpol)

            trajectory.update(x, y, speed=speed)

            # time elapsed
            _time = (
                np.arange(len(trajectory.speed)) / n_interpolated_frames / 60
            )

            # plot the arena
            draw.ROI(ROI, set_ax=True, shade=False, ax=axes["A"])

            # plot tracking so far
            draw.Tracking(trajectory.x, trajectory.y, lw=3, ax=axes["A"])
            draw.Dot(x, y, s=50, ax=axes["A"])

            # plot speed and velocity vectors
            draw.Arrow(
                x,
                y,
                interpolate_at_frame(path.velocity.angle, frame, interpol),
                L=1
                / 60
                * 2
                * interpolate_at_frame(
                    path.velocity.magnitude, frame, interpol
                ),
                color=blue,
                outline=True,
                width=2,
                ax=axes["A"],
            )
            draw.Arrow(
                x,
                y,
                path.acceleration.angle[frame],
                L=1 / 60 * 2 * path.acceleration.magnitude[frame],
                color=pink,
                outline=True,
                width=2,
                ax=axes["A"],
                zorder=200,
            )

            # plot speed trace
            axes["B"].fill_between(
                _time, 0, trajectory.speed, alpha=0.5, color=colors.speed
            )
            axes["B"].plot(_time, trajectory.speed, lw=4, color=colors.speed)

            axes["A"].set(title=f"{ROI} - crossing: {crossing_id}")
            axes["B"].set(xlabel="time (s)", ylabel="speed (cm/s)")
            camera.snap()
        # if frame > 10:
        #     break

    # ------------------------------- video creation ------------------------------- #
    print("Saving")
    animation = camera.animate(interval=1000 / FPS)
    animation.save(save_folder / f"{ROI}_{crossing_id}.mp4", fps=FPS)
    print("Done!")


def animate_all(ROI: str, FPS: int):
    """
        Creates an amimation for each bout in a ROI
    """
    bouts = pd.read_hdf(
        f"/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/roi_crossings/{ROI}_crossings.h5"
    )

    for bout_n in range(len(bouts)):
        animate_roi_crossing(ROI, bout_n, FPS)


if __name__ == "__main__":
    ROI = "T1"
    FPS = 10

    animate_all(ROI, FPS)
