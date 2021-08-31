import pandas as pd
from dataclasses import dataclass
import numpy as np
from collections import namedtuple


@dataclass
class TrackingData:
    bp: str
    x: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    direction_of_movement: np.ndarray
    orientation: np.ndarray = None
    angular_velocity: np.ndarray = None
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
