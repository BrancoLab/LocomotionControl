from dataclasses import dataclass
import numpy as np

from fcutils.maths import derivative


@dataclass
class Waypoint:
    x: int
    y: int
    theta: int
    speed: float = 0
    accel: float = 0
    segment_duration: float = 0  # time elapsed since last segment


class Waypoints:
    """
        Position and angle of a series of waypoints on the Hairpin maze
    """

    waypoints_dubin = [
        Waypoint(20, 35, 270, 0),  # start
        Waypoint(20, 28, 280, 10),  # start
        Waypoint(22, 12, 270, 15),  # 1st bend
        Waypoint(17, 2, 180, 5),  # 1st bend
        Waypoint(11, 7, 95, 7),  # 1st bend
        Waypoint(12, 36, 80, 35),  # 2nd bend
        Waypoint(20, 47, 0, 20),  # 2nd bend - halfway
        Waypoint(28, 42, 290, 30),  #
        Waypoint(30, 35, 270, 33),  #
        Waypoint(28, 12, 270, 28),  # 3nd bend
        Waypoint(33, 3, 0, 10),  # 3nd bend
        Waypoint(38.75, 9, 80, 14),  # 3nd bend
        Waypoint(38, 42, 100, 43),  # 4nd bend
        Waypoint(35, 49, 125, 38),  # 4nd bend - halfway
        Waypoint(27, 55, 180, 30),  # 4nd bend - halfway
        Waypoint(15, 55, 180, 30),  # 4nd bend - halfway
        Waypoint(6, 51, 240, 25),  # 4nd bend - halfway
        Waypoint(3, 40, 270, 30),  # 4nd bend
        Waypoint(3, 7, 270, 0),  # end
    ]

    waypoints_spline = [
        Waypoint(20, 35, 270, 0),  # start
        Waypoint(20, 35, 270, 0),  # start
        Waypoint(20, 28, 280, 15),  # start
        Waypoint(22, 12, 270, 5),  # 1st bend
        Waypoint(19.5, 3, 180, 10),  # 1st bend
        Waypoint(11, 7, 95, 35),  # 1st bend
        Waypoint(11, 36, 80, 10),  # 2nd bend
        Waypoint(20, 47, 0, 30),  # 2nd bend - halfway
        Waypoint(30, 35, 270, 40),  #
        Waypoint(28, 12, 270, 40),  # 3nd bend
        Waypoint(32, 3, 0, 15),  # 3nd bend
        Waypoint(39, 10, 80, 20),  # 3nd bend
        Waypoint(38, 40, 100, 45),  # 4nd bend
        Waypoint(27, 55, 180, 30),  # 4nd bend - halfway
        Waypoint(5, 50, 240, 12),  # 4nd bend
        Waypoint(3, 26, 270, 5),  # end
        Waypoint(3, 7, 270, 0),  # end
        Waypoint(3, 7, 270, 0),  # end
    ]

    def __init__(self, use: str = "dubin", waypoints: list = None):
        if waypoints is None:
            if use == "dubin" or use == "quintic":
                self.waypoints = self.waypoints_dubin
            elif use == "spline":
                self.waypoints = self.waypoints_spline
            else:
                raise ValueError(use)
        else:
            self.waypoints = waypoints

    _idx = 0

    @classmethod
    def from_tracking(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        speed: np.ndarray,
        theta: np.ndarray,
        sample_every: int = 20,
    ):
        accel = derivative(speed)
        waypoints = []
        for frame in np.arange(0, len(x), sample_every):
            angle = theta[frame] + 90
            if angle < 0:
                angle += 360
            elif angle > 360:
                angle -= 360

            waypoints.append(
                Waypoint(
                    x[frame],
                    y[frame],
                    angle,
                    speed[frame],
                    accel[frame],
                    segment_duration=20 / 60,
                )
            )
        return cls(waypoints=waypoints)

    @classmethod
    def from_list(cls, wps: list):
        waypoints = Waypoints()
        waypoints.waypoints = wps
        return waypoints

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
