import sys
import pandas as pd
import matplotlib.pyplot as plt
from celluloid import Camera
import numpy as np
from loguru import logger

sys.path.append("./")


from fcutils.progress import track
from tpd import recorder

from data import colors, paths

# from data.dbase.db_tables import ROICrossing
import draw
from geometry import Path, GrowingPath
from geometry.interpolate import interpolate_at_frame
from analysis.ROI.roi_plots import plot_roi_crossing


"""
    Creates an animation of a single ROI crossing showin the position of the mouse and 
    velocity/acceleration vectors
"""

# TODO add body tracking once it's of good enough quality
# TODO save images with whole bout


def animate_one(ROI: str, crossing_id: int, FPS: int):
    scale = 1 / 30  # to scale velocity vectors
    save_folder = (
        paths.analysis_folder / "behavior" / "roi_crossings_animations"
    )

    # ----------------------------------- prep ----------------------------------- #
    # load tracking
    bouts = pd.read_hdf(
        paths.analysis_folder
        / "behavior"
        / "roi_crossings"
        / f"{ROI}_crossings.h5"
    )
    bout = bouts.sort_values("duration").iloc[crossing_id]
    logger.info(f"Animating bout {crossing_id} - {round(bout.duration, 2)}s")
    # tracking: dict = ROICrossing.get_crossing_tracking(bout.crossing_id)

    # generate a path from tracking
    path = Path(bout.x, bout.y)

    # create fig and start camera
    if ROI == "T4":
        f = plt.figure(figsize=(12, 12))
    elif "S" in ROI:
        f = plt.figure(figsize=(8, 14))
    else:
        f = plt.figure(figsize=(5, 14))

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
                interpolate_at_frame(
                    path.velocity.angle, frame, interpol, method="step"
                ),
                L=scale
                * interpolate_at_frame(
                    path.velocity.magnitude, frame, interpol
                ),
                color=colors.velocity,
                outline=True,
                width=2,
                ax=axes["A"],
            )
            draw.Arrow(
                x,
                y,
                path.acceleration.angle[frame],
                L=4 * scale * path.acceleration.magnitude[frame],
                color=colors.acceleration,
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

    # ------------------------------- video creation ------------------------------- #
    print("Saving")
    animation = camera.animate(interval=1000 / FPS)
    animation.save(save_folder / f"{ROI}_{bout.crossing_id}.mp4", fps=FPS)
    print("Done!")

    plt.cla()
    plt.close(f)
    del camera
    del animation

    # ----------------------------------- plot ----------------------------------- #
    f = plot_roi_crossing(bout, step=4)
    recorder.add_figures(svg=False)
    plt.close(f)


def animate_all(ROI: str, FPS: int):
    """
        Creates an amimation for each bout in a ROI
    """
    bouts = pd.read_hdf(
        paths.analysis_folder
        / "behavior"
        / "roi_crossings"
        / f"{ROI}_crossings.h5"
    )

    for bout_n in range(len(bouts)):
        animate_one(ROI, bout_n, FPS)


if __name__ == "__main__":
    recorder.start(
        paths.analysis_folder / "behavior",
        "roi_crossings_animations",
        timestamp=False,
    )

    ROI = "S2"
    FPS = 100
    BOUT_ID = 0

    animate_one(ROI, BOUT_ID, FPS)
    # animate_all(ROI, FPS)
    recorder.describe()
