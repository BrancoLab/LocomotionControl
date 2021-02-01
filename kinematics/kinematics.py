import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from collections import namedtuple
from celluloid import Camera
from matplotlib.patches import Polygon


from fcutils.maths.signals import rolling_mean, derivative
from fcutils.progress import track

from myterial import (
    salmon,
    salmon_darker,
    indigo,
    indigo_darker,
    blue_grey_darker,
    green_dark,
    teal,
    teal_dark,
)

PAW_COLORS = dict(
    HL=salmon,
    FL=indigo,
    HR=salmon_darker,
    FR=indigo_darker,
    L=salmon,
    R=indigo,
    body=teal,
    snout=teal_dark,
    B=teal_dark,
    tail_base=green_dark,
    B2=green_dark,
    right_ear=green_dark,
    left_ear=green_dark,
)

BP = namedtuple("bp", "x, y")


class BodyPart:
    def __init__(self, tracking, bpname, cm_per_px, fps):
        """
            Represents tracking data of a single body part
        """
        self.x = rolling_mean(tracking[f"{bpname}_x"].values, 5) * cm_per_px
        self.y = rolling_mean(tracking[f"{bpname}_y"].values, 5) * cm_per_px
        self.speed = (
            rolling_mean(tracking[f"{bpname}_speed"].values, 5)
            * cm_per_px
            * fps
        )

    def to_egocentric(self, frame, T, R):
        """
            Transforms the body parts coordinates from allocentric
            to egocentric (wrt to the body's position and orientation)

            Arguments:
                frame: int. Frame number
                T: np.ndarray. Transform matrix to convert allo -> ego
                R: np.ndarray. Transform matrix to remove rotations of body axis
        """
        point = np.array([self.x[frame], self.y[frame]])
        ego_point = R @ (point + T)
        return ego_point


class Kinematics:
    cm_per_px = 1 / 30.8
    fps = 60

    def __init__(self, tracking_data_path):
        """
            Represents the tracking data of multiple body parts and 
            provides methods for applying transforms to data (e.g. going
            from allocentric to egocentric representations)
        """
        # load tracking data from a .h5 file
        tracking = pd.read_hdf(tracking_data_path, key="hdf")
        self.n_frames = len(tracking)

        self.HL = BodyPart(tracking, "left_hl", self.cm_per_px, self.fps)
        self.FL = BodyPart(tracking, "left_fl", self.cm_per_px, self.fps)
        self.HR = BodyPart(tracking, "right_hl", self.cm_per_px, self.fps)
        self.FR = BodyPart(tracking, "right_fl", self.cm_per_px, self.fps)
        self.left_ear = BodyPart(
            tracking, "left_ear", self.cm_per_px, self.fps
        )
        self.right_ear = BodyPart(
            tracking, "right_ear", self.cm_per_px, self.fps
        )
        self.snout = BodyPart(tracking, "snout", self.cm_per_px, self.fps)
        self.body = BodyPart(tracking, "body", self.cm_per_px, self.fps)
        self.tail_base = BodyPart(
            tracking, "tail_base", self.cm_per_px, self.fps
        )

        self.bps = dict(
            HL=self.HL,
            FL=self.FL,
            left_ear=self.left_ear,
            snout=self.snout,
            right_ear=self.right_ear,
            FR=self.FR,
            HR=self.HR,
            body=self.body,
            tail_base=self.tail_base,
        )
        self.segments = dict(L=("HL", "FR"), R=("HR", "FL"),)

        theta = np.degrees(
            np.unwrap(np.radians(tracking["body_lower_bone_orientation"]))
        )
        self.orientation = rolling_mean(theta, 5)

        self.v = self.body.speed
        self.omega = derivative(self.orientation) * self.fps

    def T(self, frame):
        """
            T represents the transform matrix to convert from
            allocentric to egocentric coordinate space at a given frame

            Arguments:
                frame: int. Frame number

            Returns:
                R: 2x1 np.array with transform matrix for translations
        """
        x, y = self.body.x[frame], self.body.y[frame]
        return np.array([-x, -y])

    def R(self, frame):
        """
            R is the transform matrix to remove rotations of the body axis. 
        """
        theta = np.radians(self.orientation[frame] + 180)
        return np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

    def bparts(self, frame, egocentric=False):
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
        if not egocentric:
            return {
                bp: BP(bpart.x[frame], bpart.y[frame])
                for bp, bpart in self.bps.items()
            }
        else:
            T = self.T(frame)
            R = self.R(frame)

            parts = {}
            for bp, bpart in self.bps.items():
                x, y = bpart.to_egocentric(frame, T, R)
                parts[bp] = BP(x, y)

            return parts

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

        head = ("snout", "right_ear", "body", "left_ear")
        body = ("FL", "FR", "HR", "tail_base", "HL")
        paws = ("FL", "FR", "HR", "HL")

        for frame in track(
            range(self.n_frames),
            total=self.n_frames,
            description="Animating...",
        ):
            bps = self.bparts(frame)
            bps_ego = self.bparts(frame, egocentric=True)

            # draw mouse
            for names, color in zip((head, body), (salmon, teal_dark)):
                for ax, _bps in zip(axarr, (bps, bps_ego)):
                    x = [_bps[name].x for name in names]
                    y = [_bps[name].y for name in names]
                    mouse = Polygon(
                        np.vstack([x, y]).T,
                        True,
                        lw=2,
                        color=color,
                        joinstyle="round",
                        edgecolor=blue_grey_darker,
                        zorder=-5,
                        alpha=1,
                    )
                    ax.add_artist(mouse)

            # draw each PAW
            for name, bp in bps.items():
                color = PAW_COLORS[name]
                if name in paws:
                    s, lw = 100, 0.5
                else:
                    s, lw = 140, 1

                axarr[0].scatter(
                    bp.x,
                    bp.y,
                    s=s,
                    color=color,
                    lw=lw,
                    edgecolors=[0.2, 0.2, 0.2],
                )
                axarr[1].scatter(
                    bps_ego[name].x,
                    bps_ego[name].y,
                    s=s,
                    color=color,
                    lw=lw,
                    edgecolors=[0.2, 0.2, 0.2],
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
    kin = Kinematics(
        "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/experimental_validation/FC_210122_BA1099282_trial_9_tracking.h5"
    )
    kin.animate(fps=20)
