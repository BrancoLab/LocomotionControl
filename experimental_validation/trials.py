import sys
import pandas as pd
from loguru import logger
import numpy as np


from myterial import orange
from fcutils.path import from_json, files
from fcutils.maths.signals import rolling_mean, derivative

sys.path.append("./")

from experimental_validation import paths
from experimental_validation._tracking import transform, unwrap, fps

from kinematics.bodypart import BodyPart
from kinematics.fixtures import BODY_PARTS_NAMES


class Trial:
    def __init__(self, name, info, tracking_path=None, get_endpoints=True):
        """
            Represents a single trial, with pointers to its metadata, files etc.

            Arguments:
                name: str. Trial name.
                info: dict. Metadata
                tracking_path: str, Path. Path to tracking data
                get_endpoints: bool. If true trial start/end are extracted from
                    tracking to get when mouse is locomoting
        """
        self.name = name
        self.session_name = name.split("_trial")[0]

        # get video paths
        self.session_video = files(paths.RAW, f"{self.session_name}_video.avi")
        self.trial_video = files(paths.TRIALS_CLIPS, f"{name}.mp4")

        # get tracking data
        tracking_path = tracking_path or files(
            paths.TRIALS_CLIPS, f"{name}_tracking.h5"
        )
        if tracking_path is not None:
            self.has_tracking = True
            self.tracking = pd.read_hdf(tracking_path, key="hdf")
        else:
            self.has_tracking = False
            self.tracking = None

        self.whole_session_tracking_path = files(
            paths.TRACKING_DATA, f"{self.session_name}_tracking.h5"
        )

        # report missing variables
        vrs = (self.session_video, self.trial_video, self.tracking)
        msgs = ("session video", "trial video", "tracking")
        for v, msg in zip(vrs, msgs):
            if v is None:
                logger.warning(f"{self.name} | {msg} was not found")

        # store metadata as attributes
        for k, v in info.items():
            setattr(self, k, v)

        # clean up tracking data
        if self.has_tracking:
            if get_endpoints:
                self.get_start_and_end()
            else:
                self.start, self.end = 0, len(self.tracking["body_speed"])
                self.good = True

            if self.good:
                self.cleanup_tracking()
                self.n_frames = len(self.body.x)
                logger.info(f"[green]{self} | loaded correctly")

    def __repr__(self):
        return f"Trial: {self.name}"

    def __str__(self):
        return f"Trial: {self.name}"

    def __rich__(self):
        return f"Trial: [{orange}]{self.name}"

    def __getitem__(self, key):
        return self.__dict__[key]

    def __len__(self):
        if self.has_tracking:
            return len(self.body.x)
        return None

    @property
    def whole_session_tracking(self):
        if self.whole_session_tracking_path is not None:
            return pd.read_hdf(self.whole_session_tracking_path, key="hdf")
        else:
            return None

    # ---------------------------------- on load --------------------------------- #

    def get_start_and_end(self):
        """
            Gets start and end frames of trial run based on when
            the mouse crosses given thresholds
        """
        y = transform(self.tracking["body_y"], start=0, scale=False)
        try:
            start = np.where(y > 1400)[0][-1]
            end = np.where(y > 300)[0][-1]

            # check if the mouse stops in this interval
            # if it stops the stop frame is the end of the run
            bspeed = (
                rolling_mean(
                    transform(
                        self.tracking["body_speed"], start=start, end=end
                    ),
                    10,
                )
                * fps
            )
            if np.any(bspeed < 5):
                end = np.where(bspeed < 5)[0][0] + start

            if end <= start:
                raise IndexError(
                    "End cannot be equal to or smaller than start"
                )
        except IndexError:
            logger.info(f"{self} | mouse didnt go below Y threshold, skipping")
            self.start, self.end = None, None
            self.good = False
        else:
            self.start, self.end = start, end

    def cleanup_tracking(self):
        """
            Cleans up and organizes the tracking data for easier use
        """
        # get each body part
        for bp in BODY_PARTS_NAMES:
            setattr(
                self,
                bp,
                BodyPart(bp, self.tracking, start=self.start, end=self.end),
            )

        # get body orientation
        self.orientation = transform(
            unwrap(self.tracking["body_lower_bone_orientation"]),
            scale=False,
            start=self.start,
            end=self.end,
        )

        # store linear and angulare velocities
        self.v = self.body.speed
        self.omega = derivative(self.orientation) * fps

        # collate speeds for easier access
        self.speeds = dict(
            left_hl=self.left_hl.speed,
            left_fl=self.left_fl.speed,
            right_hl=self.right_hl.speed,
            right_fl=self.right_fl.speed,
        )

    # -------------------------------- kinematics -------------------------------- #


class Trials:
    def __init__(self, only_tracked=False):
        """
            Loads and stores all trials and metadata
        """
        self.trecords = from_json(paths.trials_records)
        n_good = len([t for t in self.trecords.values() if t["good"]])
        logger.info(
            f"Found metadata for {len(self.trecords)} trials [{n_good} good ones]"
        )

        self.trials = [
            Trial(name, info)
            for name, info in self.trecords.items()
            if info["good"]
        ]

        if only_tracked:
            self.trials = [t for t in self.trials if t.has_tracking and t.good]

        logger.debug(f"Loaded {len(self.trials)} trials")
        self._idx = 0

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, item):
        return self.trials[item]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            nxt = self.trials[self._idx]
        except IndexError:
            raise StopIteration
        else:
            self._idx += 1
            return nxt

    def __repr__(self):
        return f"{len(self)} trials"

    def __str__(self):
        return f"{len(self)} trials"


if __name__ == "__main__":
    trials = Trials()
    print(trials[0])
