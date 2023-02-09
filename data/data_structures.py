import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from collections import namedtuple

from typing import Optional

import sys

sys.path.append("./")


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
    u: np.ndarray = None
    udot: np.ndarray = None
    beta: np.ndarray = None

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
class AtPoint:
    """
        Class used to collect kinematics data across different Locomotion traces, for each
        trace it stores the locomotion data for a single time point (i.e. when the mouse is at
        a selected point in the arena). Useful to e.g. compare kinematics everytime the mice are
        at the apex of a turn.
    """

    color: str
    name: str
    frame_idx: list = field(
        default_factory=list
    )  # index of the selected frame of each locomotion
    G: Optional[float] = None  # g_coord value
    locomotions: list = field(default_factory=list)
    track_distance: list = field(
        default_factory=list
    )  # distance along linearized track at frame
    G_distance: list = field(default_factory=list)  # gcoord at selected frame

    @property
    def x(self) -> np.ndarray:
        return np.array([loc.body.x for loc in self.locomotions])

    @property
    def y(self) -> np.ndarray:
        return np.array([loc.body.y for loc in self.locomotions])

    @property
    def speed(self) -> np.ndarray:
        return np.array([loc.body.speed for loc in self.locomotions])

    def add(
        self, locomotion, frame: int, center_line=None,
    ):
        """
            Adds a locomotion trace data to the class' data
        """
        # get locomotion at frame
        at_frame = locomotion @ frame

        self.locomotions.append(at_frame)
        self.frame_idx.append(frame)

        # project to linearized track
        # if center_line is not None:
        #     self.track_distance.append(
        #         TCS.point_to_track_coordinates_system(
        #             center_line, (at_frame.body.x, at_frame.body.y)
        #         )[0]
        #     )

        # get the G_coordinates
        self.G_distance.append(locomotion.gcoord[frame])
