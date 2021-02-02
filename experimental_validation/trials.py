import sys
import pandas as pd
from loguru import logger
import numpy as np


from myterial import orange
from fcutils.path import from_json, files
from fcutils.maths.signals import rolling_mean, derivative

sys.path.append("./")

from experimental_validation import paths
from experimental_validation._tracking import transform, unwrap, cm_per_px, fps
from experimental_validation._steps_utils import (
    get_paw_steps_times,
    get_diagonal_steps,
    step_times,
)


class BodyPart:
    def __init__(self, bpname, tracking=None, start=0, end=-1):
        """
            Represents tracking data of a single body part
        """
        self.name = bpname

        if tracking is not None:
            self.x = (
                rolling_mean(tracking[f"{bpname}_x"].values, 5) * cm_per_px
            )
            self.y = (
                rolling_mean(tracking[f"{bpname}_y"].values, 5) * cm_per_px
            )
            self.speed = (
                rolling_mean(tracking[f"{bpname}_speed"].values, 5)
                * cm_per_px
                * fps
            )

            # tr uncate
            self.x = self.x[start:end]
            self.y = self.y[start:end]
            self.speed = self.speed[start:end]

    @classmethod
    def from_data(cls, name, x, y, speed):
        """
            Instantiate from pre processed data
        """
        new = cls(name)
        new.x = x
        new.y = y
        new.speed = speed
        return new

    def truncate(self, start, end):
        """
            Returns a copy of the tracking data truncated
            between two frames
        """
        x = self.x.copy()[start:end]
        y = self.y.copy()[start:end]
        speed = self.speed.copy()[start:end]

        return BodyPart.from_data(self.name, x, y, speed)

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


class Trial:
    bp_names = (
        "snout",
        "right_ear",
        "right_fl",
        "right_hl",
        "tail_base",
        "left_hl",
        "left_fl",
        "left_ear",
        "body",
    )
    head_names = ("snout", "right_ear", "body", "left_ear")
    paws_names = ("right_fl", "right_hl", "left_hl", "left_fl")
    body_names = ("right_fl", "right_hl", "tail_base", "left_hl", "left_fl")
    body_axis_names = ("snout", "body", "tail_base")

    def __init__(self, name, info):
        """
            Represents a single trial, with pointers to its metadata, files etc.

            Arguments:
                name: str. Trial name.
                info: dict. Metadata
        """
        self.name = name
        self.session_name = name.split("_trial")[0]

        # get video paths
        self.session_video = files(paths.RAW, f"{self.session_name}_video.avi")
        self.trial_video = files(paths.TRIALS_CLIPS, f"{name}.mp4")

        # get tracking data
        tracking_path = files(paths.TRIALS_CLIPS, f"{name}_tracking.h5")
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
            self.get_start_and_end()
            self.cleanup_tracking()
            self.n_frames = len(self.body.x)

    def __repr__(self):
        return f"Trial: {self.name}"

    def __str__(self):
        return f"Trial: {self.name}"

    def __rich__(self):
        return f"Trial: [{orange}]{self.name}"

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
                transform(self.tracking["body_speed"], start=start, end=end)
                * fps
            )
            if np.any(bspeed < 5):
                end = np.where(bspeed < 5)[0][0] + start
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
        if not self.good:
            return

        # get each body part
        for bp in self.bp_names:
            setattr(
                self,
                bp,
                BodyPart(bp, self.tracking, start=self.start, end=self.end),
            )

        # get body orientation
        self.orientation = transform(
            unwrap(self.tracking["body_whole_bone_orientation"]),
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

    def extract_steps(
        self, step_speed_th, precise_th=12,
    ):
        """ 
            Given speed traces for the four limbs, this function returns the 
            steps the animal took, in particular it looks for 'diagonal' steps
            (when the animal is in trot) involving pairs of diagonally connected limbs

            Arguments:
                step_speed_th: float, initial threshold for step onset/offset (swing phase)
                precise_th: float, second threshold for more accurately identifying the
                    onset and offset of steps

            Returns:
                LH/LF/RH/RF_steps: step_times named tuples with onset/offset of steps
                    for each limb
                diagonal_steps: step_times named tuples with onset/offset of diagonal steps
                diag_data: dictionary with data about each diagonal step (e.g. which side, based
                    on hind limb)
                step_starts: 1d numpy array with time of each RH diagonal step, useful for plotting.

        """
        # get steps for each paw and diagonal
        LH_steps = get_paw_steps_times(
            self.left_hl.speed, step_speed_th, precise_th=precise_th
        )
        RF_steps = get_paw_steps_times(
            self.right_fl.speed, step_speed_th, precise_th=precise_th
        )
        L_diagonal_steps, L_diag_data = get_diagonal_steps(
            LH_steps, RF_steps, self.left_hl.speed, self.right_fl.speed
        )

        RH_steps = get_paw_steps_times(
            self.right_hl.speed, step_speed_th, precise_th=precise_th
        )
        LF_steps = get_paw_steps_times(
            self.left_fl.speed, step_speed_th, precise_th=precise_th
        )
        R_diagonal_steps, R_diag_data = get_diagonal_steps(
            RH_steps, LF_steps, self.right_hl.speed, self.left_fl.speed
        )

        # merge R/L diagonal steps to get complete step
        _starts = np.concatenate(
            (R_diagonal_steps.starts, L_diagonal_steps.starts)
        )
        _names = np.concatenate(
            [
                [
                    "R" + str(x + 1)
                    for x in np.arange(len(R_diagonal_steps.starts))
                ],
                [
                    "L" + str(x + 1)
                    for x in np.arange(len(L_diagonal_steps.starts))
                ],
            ]
        )
        _names = _names[np.argsort(_starts)]

        pooled_starts, pooled_ends, diag_data = [], [], {}
        count = 0
        for n in _names:
            side, idx = n[0], int(n[1])
            if side == "R":
                pooled_starts.append(R_diagonal_steps.starts[idx - 1])
                pooled_ends.append(R_diagonal_steps.ends[idx - 1])
                data = R_diag_data[idx - 1]
                data["paws"] = ("right_hl", "left_fl")
            else:
                pooled_starts.append(L_diagonal_steps.starts[idx - 1])
                pooled_ends.append(L_diagonal_steps.ends[idx - 1])
                data = L_diag_data[idx - 1]
                data["paws"] = ("left_hl", "right_fl")

            data["side"] = side
            diag_data[count] = data
            count += 1
        diagonal_steps = step_times(pooled_starts, pooled_ends)

        # step starts
        step_starts = (
            np.array(R_diagonal_steps.starts) + self.start
        )  # to mark the start of each L-R step sequence

        return (
            LH_steps,
            RF_steps,
            LF_steps,
            RH_steps,
            R_diagonal_steps,
            L_diagonal_steps,
            diagonal_steps,
            diag_data,
            step_starts,
        )


class Trials:
    def __init__(self):
        """
            Loads and stores all trials and metadata
        """
        self.trecords = from_json(paths.trials_records)

        self.trials = [
            Trial(name, info)
            for name, info in self.trecords.items()
            if info["good"]
        ]
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
