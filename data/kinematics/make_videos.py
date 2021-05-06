import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from celluloid import Camera
from matplotlib.patches import Polygon
import sys
from PIL import Image


from fcutils.maths import coordinates
from fcutils.progress import track
from fcutils.video import get_cap_from_file, get_cap_selected_frame

from myterial import (
    blue_grey_darker,
    blue_grey,
)

sys.path.append("./")
from experimental_validation._tracking import cm_per_px
from experimental_validation.trials import Trials
from experimental_validation.paths import analysis_folder

from kinematics.fixtures import (
    BODY_PARTS_NAMES,
    BODY_PARTS_COLORS,
    HEAD_NAMES,
    BODY_NAMES,
    PAWS_NAMES,
)
from kinematics.bodypart import BodyPart
from kinematics.plot_utils import point, line


class Kinematics:
    segments = dict(
        left_paws=("left_hl", "right_fl"), right_paws=("right_hl", "left_fl"),
    )

    def __init__(self, trial):
        """
            Represents the tracking data of multiple body parts and 
            provides methods for applying transforms to data (e.g. going
            from allocentric to egocentric representations)

            Arguments
                trial:
                    instance of Trial class
        """
        if not trial.has_tracking:
            raise ValueError(f"{trial} has no tracking")

        self.trial = trial

        # get the video
        self.video = get_cap_from_file(self.trial.trial_video)
        self.save_path = analysis_folder / f"{self.trial.name}_kinematics.mp4"

    def T(self, frame):
        """
            T represents the transform matrix to convert from
            allocentric to egocentric coordinate space at a given frame

            Arguments:
                frame: int. Frame number

            Returns:
                R: 2x1 np.array with transform matrix for translations
        """
        x, y = self.trial.body.x[frame], self.trial.body.y[frame]
        return np.array([-x, -y])

    def R(self, frame):
        """
            R is the transform matrix to remove rotations of the body axis. 
        """
        return coordinates.R(self.trial.orientation[frame] + 180)

    def get_video_frames(self, frame):
        """
            Gets the video frame from the trial video and rotates/crops it
            to align it with the egocentric view

            Arguments:
                frame: int. Frame number

            Returns:
                whole_frame: np.ndarray with whole frame
                ego_frame: np.ndarray with frame for egocentric view
        """
        whole_frame = get_cap_selected_frame(self.video, frame)

        # crop and rotate frame for egocentric view
        ego_frame = Image.fromarray(whole_frame)

        cut = int(10 * 1 / cm_per_px)
        x = int(self.trial.body.x[frame - self.trial.start] * 1 / cm_per_px)
        y = int(self.trial.body.y[frame - self.trial.start] * 1 / cm_per_px)
        ego_frame = ego_frame.crop(box=(x - cut, y - cut, x + cut, y + cut))

        ego_frame = ego_frame.rotate(
            -self.trial.orientation[frame - self.trial.start] + 180
        )

        return whole_frame, ego_frame

    def bparts_positions(self, frame, egocentric=False):
        """
            Returns the position of all body parts at a given 
            frame.

            Arguments:
                frame: int. Frame number
                egocentric: bool. If true the positions are
                    in egocentric coordinates
            
            Returns:
                bp: dict of body parts coordintes
        """
        # get transform matrices for egocentric
        T = self.T(frame)
        R = self.R(frame)

        # get bps positions
        positions = {}
        for bp in BODY_PARTS_NAMES:
            bpart = getattr(self.trial, bp)

            if not egocentric:
                positions[bp] = bpart.at_frame(frame)
            else:
                x, y = bpart.to_egocentric(frame, T, R)
                positions[bp] = BodyPart.from_data(
                    bp, x, y, np.nan
                )  # ! speed is not adjusted!
        return positions

    def animate(self, fps=60):
        """
            Creates an animation with the tracking data

            Argument:
                fps = int. Fps of animation
        """
        if self.save_path.exists():
            logger.info(
                f'Video for trial "{trial.name}" exists already, skipping.'
            )
            return

        logger.info(
            f"Creating animation: {self.trial.n_frames} frames for : '{self.trial.name}'"
        )
        f, axarr = plt.subplots(ncols=2, figsize=(12, 8))
        camera = Camera(f)

        axarr[0].set(xlim=[0, 45], ylim=[0, 65])
        axarr[1].set(xlim=[-10, 10], ylim=[-10, 10])

        for frame in track(
            range(self.trial.n_frames),
            total=self.trial.n_frames,
            description="Animating...",
            transient=True,
        ):
            # get body parts position
            bps = self.bparts_positions(frame)
            bps_ego = self.bparts_positions(frame, egocentric=True)

            # draw video frames
            whole, rotated = self.get_video_frames(frame + self.trial.start)
            axarr[0].imshow(whole, origin="lower", extent=(0, 45, 0, 65))
            axarr[1].imshow(rotated, origin="lower", extent=(-10, 10, -10, 10))

            # draw mouse
            _names = (HEAD_NAMES, BODY_NAMES)
            colors = (blue_grey, blue_grey_darker)
            for names, color in zip(_names, colors):
                for ax, _bps in zip(axarr, (bps, bps_ego)):
                    x = [_bps[name].x for name in names]
                    y = [_bps[name].y for name in names]
                    mouse = Polygon(
                        np.vstack([x, y]).T,
                        True,
                        lw=0,
                        color=color,
                        joinstyle="round",
                        zorder=-5,
                        alpha=0.5,
                    )
                    ax.add_artist(mouse)

            # draw each PAW
            for name, bp in bps.items():
                color = BODY_PARTS_COLORS[name]
                if name in PAWS_NAMES or name == "body":
                    s, lw, alpha = 160, 1, 1
                else:
                    s, lw, alpha = 80, 0.4, 0.8

                point(
                    bp,
                    axarr[0],
                    s=int(s / 3),
                    color=color,
                    lw=lw,
                    edgecolors=[0.2, 0.2, 0.2],
                    alpha=alpha,
                )
                point(
                    bps_ego[name],
                    axarr[1],
                    s=s,
                    color=color,
                    lw=lw,
                    edgecolors=[0.2, 0.2, 0.2],
                    alpha=alpha,
                )

            # draw lines between paws
            for side, (bp1, bp2) in self.segments.items():
                if side in ("B", "B2"):
                    lw = 5
                else:
                    lw = 3

                line(
                    bps[bp1],
                    bps[bp2],
                    axarr[0],
                    lw=lw,
                    zorder=100,
                    color=BODY_PARTS_COLORS[side],
                )
                line(
                    bps_ego[bp1],
                    bps_ego[bp2],
                    axarr[1],
                    lw=lw,
                    zorder=100,
                    color=BODY_PARTS_COLORS[side],
                )

            camera.snap()

        logger.info("Saving animation at video.mp4")
        interval = int(np.ceil(1000 / fps))
        camera.animate(interval=interval).save(str(self.save_path))
        logger.debug("done")


if __name__ == "__main__":
    for trial in track(Trials(only_tracked=True)):
        kin = Kinematics(trial)
        kin.animate(fps=20)
