import sys
import numpy as np

from fcutils.maths.signals import rolling_mean

sys.path.append("./")

from experimental_validation._tracking import cm_per_px, fps


class BodyPart:
    def __init__(
        self, bpname, tracking=None, start=0, end=-1, smooth_window=5
    ):
        """
            Represents tracking data of a single body part
        """
        self.name = bpname

        if tracking is not None:
            self.x = tracking[f"{bpname}_x"].values * cm_per_px
            self.y = tracking[f"{bpname}_y"].values * cm_per_px
            self.speed = (
                rolling_mean(tracking[f"{bpname}_speed"].values, 5)
                * cm_per_px
                * fps
            )

            # tr uncate
            self.x = self.x[start:end]
            self.y = self.y[start:end]
            self.speed = self.speed[start:end]

    def __repr__(self):
        return "BodyPart: " + self.name + f" [{len(self)}] "

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.x)

    @property
    def xy(self):
        return np.vstack([self.x, self.y])

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

    def at_frame(self, frame):
        """
            Returns a copy of itself with only data for a given
            frame

            Arguments:
                frame: int. Frame number
        """
        return BodyPart.from_data(
            self.name, self.x[frame], self.y[frame], self.speed[frame]
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
