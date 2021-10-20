from dataclasses import dataclass
import numpy as np


@dataclass
class Waypoint:
    x: int
    y: int
    theta: int


class Waypoints:
    """
        Position and angle of a series of waypoints on the Hairpin maze
    """

    waypoints_dubin = [
        Waypoint(20, 35, 270),  # start
        Waypoint(20, 28, 280),  # start
        Waypoint(22, 12, 270),  # 1st bend
        Waypoint(17, 2, 180),  # 1st bend
        Waypoint(11, 7, 95),  # 1st bend
        Waypoint(12, 36, 80),  # 2nd bend
        Waypoint(20, 47, 0),  # 2nd bend - halfway
        Waypoint(28, 42, 290),  #
        Waypoint(30, 35, 270),  #
        Waypoint(28, 12, 270),  # 3nd bend
        Waypoint(33, 3, 0),  # 3nd bend
        Waypoint(38.75, 9, 80),  # 3nd bend
        Waypoint(38, 42, 100),  # 4nd bend
        Waypoint(35, 49, 125),  # 4nd bend - halfway
        Waypoint(27, 55, 180),  # 4nd bend - halfway
        Waypoint(15, 55, 180),  # 4nd bend - halfway
        Waypoint(6, 51, 240),  # 4nd bend - halfway
        Waypoint(3, 40, 270),  # 4nd bend
        Waypoint(3, 7, 270),  # end
    ]

    waypoints_spline = [
        Waypoint(20, 35, 270),  # start
        Waypoint(20, 35, 270),  # start
        Waypoint(20, 28, 280),  # start
        Waypoint(22, 12, 270),  # 1st bend
        Waypoint(19.5, 3, 180),  # 1st bend
        Waypoint(11, 7, 95),  # 1st bend
        Waypoint(11, 36, 80),  # 2nd bend
        Waypoint(20, 47, 0),  # 2nd bend - halfway
        Waypoint(30, 35, 270),  #
        Waypoint(28, 12, 270),  # 3nd bend
        Waypoint(32, 3, 0),  # 3nd bend
        Waypoint(39, 10, 80),  # 3nd bend
        Waypoint(38, 40, 100),  # 4nd bend
        Waypoint(27, 55, 180),  # 4nd bend - halfway
        Waypoint(5, 50, 240),  # 4nd bend
        Waypoint(3, 26, 270),  # end
        Waypoint(3, 7, 270),  # end
        Waypoint(3, 7, 270),  # end
    ]

    def __init__(self, use: str = "dubin"):
        if use == "dubin":
            self.waypoints = self.waypoints_dubin
        elif use == "spline":
            self.waypoints = self.waypoints_spline
        else:
            raise ValueError(use)

    _idx = 0

    @property
    def x(self) -> np.ndarray:
        return np.array([wp.x for wp in self.waypoints])

    @property
    def y(self) -> np.ndarray:
        return np.array([wp.y for wp in self.waypoints])

    @property
    def theta(self) -> np.ndarray:
        return np.array([wp.theta for wp in self.waypoints])

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < len(self.waypoints):
            self._idx += 1
            return self.waypoints[self._idx - 1]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.waypoints)

    def __getitem__(self, item):
        return self.waypoints[item]
