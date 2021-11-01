import sys

sys.path.append("./")

from sympy import Symbol, Eq, solve
import pandas as pd
import numpy as np
from typing import Tuple

from geometry import Path
from data.data_structures import LocomotionBout


"""
    Implementation of a Minimum Squared Derivative (MSD) model for XY trajectories
    given initial and final constrints on a trajectory.

    When using the 3rd derivative the model produces a XY trajectory that minimizes the 
    jerk given initial/final constraints on position, velocity and acceleration. 
    The minimal jerk model works for human locomotion.

    This code implements the maths from: Pham et al 2017 (human locomotion paper) which
    is inspired by work from Tovote on hand movements and control theory.
"""


class MSDUnivar:
    def __init__(
        self,
        x_0: float,
        x_1: float,
        v_0: float,
        v_1: float,
        a_0: float,
        a_1: float,
    ):
        """
            Given a set of initial constraints it fits MSD to a single variable
        """

        # define weights of the polynomial
        w_0 = Symbol("w_0")
        w_1 = Symbol("w_1")
        w_2 = Symbol("w_2")
        w_3 = Symbol("w_3")
        w_4 = Symbol("w_4")
        w_5 = Symbol("w_5")

        # setup sys of equations to solve for weights (see appendinx of Pham et al)
        eqs = [
            Eq(w_0, x_0),
            Eq(w_5 + w_4 + w_3 + w_2 + w_1 + w_0, x_1),
            Eq(w_1, v_0),
            Eq(5 * w_5 + 4 * w_4 + 3 * w_3 + 2 * w_2 + w_1, v_1),
            Eq(2 * w_2, a_0),
            Eq(20 * w_5 + 12 * w_4 + 6 * w_3, a_1),
        ]

        # solve
        self._solution = solve(eqs, [w_0, w_1, w_2, w_3, w_4, w_5])
        self._weights = {str(w): w for w in [w_0, w_1, w_2, w_3, w_4, w_5]}
        # print('The solution is:\n   '+'\n   '.join([f'{s} = {res[s]}' for s in weights]))

    def __call__(self, t: float):
        return (
            self._solution[self._weights["w_5"]] * t ** 5
            + self._solution[self._weights["w_4"]] * t ** 4
            + self._solution[self._weights["w_3"]] * t ** 3
            + self._solution[self._weights["w_2"]] * t ** 2
            + self._solution[self._weights["w_1"]] * t
            + self._solution[self._weights["w_0"]]
        )


class MSD:
    def __init__(
        self,
        path: LocomotionBout,
        skip: int = 2,
        start_frame: int = 0,
        end_frame: int = -1,
    ):
        """
            Fitting the minimum jerk model for one variable (e.g., X) requires
            6 constraints and produces a 5th degree polynomial to be solved
            for time in range [0,1].

            If the entire locomotor bout is too long, use start/end times to select
            frames ragnes
        """
        self.skip = skip
        self.path = path
        self.start_frame = start_frame
        self.end_frame = end_frame

        # fit to data
        self.fit()

    def _get_constraints(self, var: str):
        start = self.start_frame + self.skip
        end = self.end_frame - self.skip

        self._start, self._end = start, end

        return (
            np.nanmedian(self.path[var][start]),  # x_0
            self.path[var][end],  # x_1
            np.nanmedian(self.path.velocity[start][var]),  # v_0
            self.path.velocity[end][var],  # v_1
            np.nanmedian(self.path.acceleration_vec[start][var]),  # a_0
            self.path.acceleration_vec[end][var],  # a_1
        )

    def fit(self):
        """
            Fitting the minimum jerk model for one variable (e.g., X) requires
            6 constraints and produces a 5th degree polynomial to be solved
            for time in range [0,1].
        """
        # fit to X and Y independently

        self.fits = {}

        for var in "xy":
            # define initial/final constraints on position, speed and acceleartion
            (x_0, x_1, v_0, v_1, a_0, a_1,) = self._get_constraints(var)

            # fit and store results
            self.fits[var] = MSDUnivar(x_0, x_1, v_0, v_1, a_0, a_1)

    def simulate(self) -> Tuple[Path, np.ndarray]:
        """
            Run the model for time t=[0, 1] so that
            it can be compared to the original data.
        """
        T = np.linspace(0, 1, len(self.path) - (self._start + self._end))

        x = np.array([self.fits["x"](t) for t in T])
        y = np.array([self.fits["y"](t) for t in T])

        # create an array to match time stamps in simulation to data
        time = (
            np.arange(len(self.path) - (self._start + self._end)) + self._start
        )

        return Path(x, y), time


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data import paths
    import draw

    # load a locomotion bout
    ROI = "T2"
    _bouts = pd.read_hdf(
        paths.analysis_folder
        / "behavior"
        / "saved_data"
        / f"{ROI}_crossings.h5"
    ).sort_values("duration")
    bout = LocomotionBout(_bouts.iloc[0])

    # fit
    msd = MSD(bout)
    trajectory, time = msd.simulate()

    # plot results
    f, axes = plt.subplots(ncols=4, figsize=(16, 9))
    draw.Tracking(bout.x, bout.y, ax=axes[0])
    draw.Tracking(trajectory.x, trajectory.y, color="salmon", ax=axes[0])

    axes[1].plot(bout.speed, color="grey")
    axes[1].plot(time, trajectory.speed, color="salmon")

    axes[2].plot(bout.acceleration, color="grey")
    axes[2].plot(time, trajectory.acceleration_mag, color="salmon")

    axes[3].plot(bout.theta, color="grey")
    axes[3].plot(time, trajectory.theta, color="salmon")

    plt.show()
