import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from celluloid import Camera
from matplotlib.patches import Polygon
import sys


from fcutils.maths import coordinates
from fcutils.progress import track
from fcutils.video import get_cap_from_file, get_cap_selected_frame

from myterial import (
    salmon,
    salmon_darker,
    indigo,
    indigo_darker,
    blue_grey_darker,
    blue_grey,
    red,
    red_darker,
    red_light,
)

sys.path.append("./")
from experimental_validation.trials import Trials, BodyPart


PAW_COLORS = dict(
    left_hl=salmon,
    left_fl=salmon_darker,
    left_ear=red,
    snout=red_darker,
    body=red,
    tail_base=red_light,
    right_hl=indigo,
    right_fl=indigo_darker,
    right_ear=red,
    left_paws="k",
    right_paws="k",
)


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

    def get_frames(self, frame):
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

        return whole_frame, None

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
        for bp in self.trial.bp_names:
            bpart = getattr(self.trial, bp)

            if not egocentric:
                positions[bp] = BodyPart.from_data(
                    bp, bpart.x[frame], bpart.y[frame], bpart.speed[frame]
                )
            else:
                x, y = bpart.to_egocentric(frame, T, R)
                positions[bp] = BodyPart.from_data(bp, x, y, np.nan)
        return positions

    def animate(self, fps=60):
        """
            Creates an animation with the tracking data

            Argument:
                fps = int. Fps of animation
        """
        logger.info("Creating animation")
        f, axarr = plt.subplots(ncols=2, figsize=(12, 8))
        camera = Camera(f)

        axarr[0].set(xlim=[0, 40], ylim=[0, 80])
        axarr[1].set(xlim=[-10, 10], ylim=[-10, 10])

        for frame in track(
            range(self.trial.n_frames),
            total=self.trial.n_frames,
            description="Animating...",
        ):
            # get body parts position
            bps = self.bparts_positions(frame)
            bps_ego = self.bparts_positions(frame, egocentric=True)

            # draw video frames
            whole, rotated = self.get_frames(frame + self.trial.start)
            axarr[0].imshow(whole, origin="upper", extent=(0, 40, 0, 80))

            # draw mouse
            _names = (self.trial.head_names, self.trial.body_names)
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
                color = PAW_COLORS[name]
                if name in self.trial.paws_names:
                    s, lw, alpha = 160, 1, 1
                else:
                    s, lw, alpha = 80, 0.4, 0.8

                axarr[0].scatter(
                    bp.x,
                    bp.y,
                    s=int(s / 3),
                    color=color,
                    lw=lw,
                    edgecolors=[0.2, 0.2, 0.2],
                    alpha=alpha,
                )
                axarr[1].scatter(
                    bps_ego[name].x,
                    bps_ego[name].y,
                    s=s,
                    color=color,
                    lw=lw,
                    edgecolors=[0.2, 0.2, 0.2],
                    alpha=alpha,
                )

            # draw lines between paws
            for side, (bp1, bp2) in self.segments.items():
                if side in ("B", "B2"):
                    lw = 10
                else:
                    lw = 5

                axarr[0].plot(
                    [bps[bp1].x, bps[bp2].x],
                    [bps[bp1].y, bps[bp2].y],
                    lw=lw,
                    zorder=-1,
                    color=PAW_COLORS[side],
                    solid_capstyle="round",
                )

                axarr[1].plot(
                    [bps_ego[bp1].x, bps_ego[bp2].x],
                    [bps_ego[bp1].y, bps_ego[bp2].y],
                    lw=lw,
                    zorder=-1,
                    color=PAW_COLORS[side],
                    solid_capstyle="round",
                )

            camera.snap()

        logger.info("Saving animation at video.mp4")
        interval = int(np.ceil(1000 / fps))
        camera.animate(interval=interval).save("video.mp4")
        logger.debug("done")


# TODO: animation add video frames
# TODO: animation make plots pretty

if __name__ == "__main__":
    trial = Trials()[-1]
    kin = Kinematics(trial)
    kin.animate(fps=4)
