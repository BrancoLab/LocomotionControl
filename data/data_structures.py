import pandas as pd
from dataclasses import dataclass
import numpy as np
from collections import namedtuple

from typing import Tuple, List

import sys

sys.path.append("./")

from geometry import Path

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


class LocomotionBout(Path):
    """
        Represents a continuous bit of locomotion in the hairpin.
        It cleans up the tracking a bit by averaging vector quantities
        over a small window.
    """

    def __init__(
        self, crossing: pd.Series, window: int = 4, linearize_to: Path = None
    ):
        self.window = window  # size of smoothing window
        super.__init__(self, crossing.x, crossing.y, crossing.theta, fps=60)
        if window:
            self.smooth(window=window)

        # extract variables from locomotion bout
        self.gcoord: np.ndarray = crossing.gcoord
        self.theta: np.ndarray = crossing.theta
        self.thetadot: np.ndarray = crossing.thetadot
        self.thetadotdot: np.ndarray = crossing.thetadotdot
        self.duration: float = crossing.duration

        self.start_frame: int = crossing.start_frame
        self.end_frame: int = crossing.end_frame

        # linearize to a reference track
        if linearize_to is not None:
            self.linearized: Path = TCS.path_to_track_coordinates_system(
                linearize_to, self
            )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item: str):
        return self.__dict__[item]

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
        A.append(bout.acceleration[start:])
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
