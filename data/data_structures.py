import pandas as pd
from dataclasses import dataclass
import numpy as np
from collections import namedtuple

from typing import Tuple, List

import sys

sys.path.append("./")

from geometry import Path, Vector
from analysis.fixtures import BPS
from kinematics import track_cordinates_system as TCS


@dataclass
class TrackingData:
    bp: str
    x: np.ndarray
    y: np.ndarray
    bp_speed: np.ndarray
    speed: np.ndarray = None
    acceleration: np.ndarray = None
    orientation: np.ndarray = None
    angular_velocity: np.ndarray = None
    angular_acceleration: np.ndarray = None
    theta: np.ndarray = None
    thetadot: np.ndarray = None
    thetadotdot: np.ndarray = None
    segment: np.ndarray = None
    global_coord: np.ndarray = None
    _columns: list = None

    @classmethod
    def from_dataframe(cls, tracking: pd.DataFrame) -> namedtuple:
        """
            Given a datraframe with tracking data for many
            body parts, return a dictionary of instances of TrackingData
        """
        columns = [
            c
            for c in list(tracking.columns)
            if c not in ["mouse_id", "name", "bpname"]
        ]
        data = {}
        for bp in tracking.bpname:
            bptracking = tracking.loc[tracking.bpname == bp].iloc[0]
            data[bp] = TrackingData(
                bp,
                **{col: bptracking[col] for col in columns},
                _columns=columns + ["bp"],
            )

        tpl = namedtuple("tracking", ", ".join(tracking.bpname))
        return tpl(*data.values())

    def to_dict(self):
        if self._columns is None:
            raise ValueError
        else:
            return {c: getattr(self, c) for c in self._columns}


@dataclass
class SnapShot:
    """
        Kinematics variables at a moment in time
    """

    x: float
    y: float
    xy: Vector
    v: Vector
    s: float
    a_mag: float
    a: Vector
    tangent: Vector
    theta: float
    thetadot: float
    thetadotdot: float


class LocomotionBout(Path):
    """
        Represents a continuous bit of locomotion in the hairpin.
        It cleans up the tracking a bit by averaging vector quantities
        over a small window.
    """

    def __init__(
        self,
        bout: dict,
        window: int = 4,
        linearize_to: Path = None,
        trim: bool = True,
        bparts_tracking: dict = None,
    ):
        self.window = window  # size of smoothing window
        super().__init__(bout["body_x"], bout["body_y"], fps=60)

        if window:
            self.smooth(window=window)

        if trim:
            fast = np.where(self.speed > 20)[0]
            self.trim_start, self.trim_end = fast[0], fast[-1]
            self.trim(self.trim_start, self.trim_end)
        else:
            self.trim_start, self.trim_end = 0, -1

        # extract variables from locomotion bout
        self.gcoord: np.ndarray = bout["gcoord"][
            self.trim_start : self.trim_end
        ]
        self.duration: float = bout["duration"] if not self.trim_end else (
            self.trim_end - self.trim_start
        ) / 60

        self.start_frame: int = bout["start_frame"] + self.trim_start
        self.end_frame: int = bout["end_frame"] if not self.trim_end else bout[
            "end_frame"
        ] - self.trim_end

        # linearize to a reference track
        if linearize_to is not None:
            self.linearized: Path = TCS.path_to_track_coordinates_system(
                linearize_to, self
            )

        # add other body parts tracking as a series of Path
        if bparts_tracking is not None:
            for bp in BPS:
                bp_path = Path(
                    bparts_tracking[bp + "_x"], bparts_tracking[bp + "_x"]
                ).smooth(window=window)
                bp_path.trim(self.trim_start, self.trim_end)
                setattr(self, bp, bp_path)

    def __repr__(self) -> str:
        return f"Bout ({self.duratoin} s)"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item: str):
        return self.__dict__[item]

    def at(self, frame: int) -> SnapShot:
        """
            Returns the kinematics variables at a frame
        """

        return SnapShot(
            x=self.x[frame],
            y=self.y[frame],
            xy=Vector(self.x[frame], self.y[frame]),
            v=self.velocity[frame],
            s=self.speed[frame],
            a_mag=self.acceleration[frame],
            a=self.acceleration_vec[frame],
            tangent=self.tangent[frame],
            theta=self.theta[frame],
            thetadot=self.thetadot[frame],
            thetadotdot=self.thetadotdot[frame],
        )

    def add_ephys(self, unit: pd.DataFrame):
        self.firing_rate = unit.firing_rate[self.start_frame : self.end_frame]
        self.spikes = unit.spikes[self.start_frame : self.end_frame]
        self.unit = unit


def merge_locomotion_bouts(bouts: List[LocomotionBout]) -> Tuple[np.ndarray]:
    """
        It concats scalar quantities across individual bouts
        X -> x pos
        Y -> y pos
        S -> speed
        A -> acceleration
        T -> theta/orientation
        AV -> angular velocity
        AA -> angular acceleration
    """
    X, Y, S, A, T, AV, AA = [], [], [], [], [], [], []

    for bout in bouts:
        start = np.where(bout.speed > 10)[0][0]
        X.append(bout.x[start:])
        Y.append(bout.y[start:])
        S.append(bout.speed[start:])
        A.append(bout.acceleration_mag[start:])
        T.append(bout.theta[start:])
        AV.append(bout.thetadot[start:])
        AA.append(bout.thetadotdot[start:])

    return (
        np.hstack(X),
        np.hstack(Y),
        np.hstack(S),
        np.hstack(A),
        np.hstack(T),
        np.hstack(AV),
        np.hstack(AA),
    )
