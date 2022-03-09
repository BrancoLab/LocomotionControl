import sys

sys.path.append("./")

from sympy import Symbol, Eq, solve
import pandas as pd
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from myterial import blue_grey

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
from loguru import logger

logger.warning(
    'See "https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/QuinticPolynomialsPlanner/quintic_polynomials_planner.py" for another implementatoin'
)


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

        self.x_0 = x_0
        self.x_1 = x_1
        self.v_0 = v_0
        self.v_1 = v_1
        self.a_0 = a_0
        self.a_1 = a_1

        # symbols
        # s_x_0 = Symbol('x_0')
        # s_x_1 = Symbol('x_1')
        # s_v_0 = Symbol('v_0')
        # s_v_1 = Symbol('v_1')
        # s_a_0 = Symbol('a_0')
        # s_a_1 = Symbol('a_1')

        # define weights of the polynomial
        w_0 = Symbol("w_0")
        w_1 = Symbol("w_1")
        w_2 = Symbol("w_2")
        w_3 = Symbol("w_3")
        w_4 = Symbol("w_4")
        w_5 = Symbol("w_5")
        t = Symbol("t")

        # solution eqn and its derivative
        x = (
            w_5 * t ** 5
            + w_4 * t ** 4
            + w_3 * t ** 3
            + w_2 * t ** 2
            + w_1 * t
            + w_0
        )
        dx = x.diff(t)
        ddx = x.diff(t).diff(t)

        # setup sys of equations to solve for weights (see appendinx of Pham et al)
        eq0 = Eq(x.subs(t, 0), x_0)
        eq1 = Eq(x.subs(t, 1), x_1)
        eq2 = Eq(dx.subs(t, 0), v_0)
        eq3 = Eq(dx.subs(t, 1), v_1)
        eq4 = Eq(ddx.subs(t, 0), a_0)
        eq5 = Eq(ddx.subs(t, 1), a_1)

        eqs = [eq0, eq1, eq2, eq3, eq4, eq5]

        # solve
        self._solution = solve(eqs, [w_0, w_1, w_2, w_3, w_4, w_5])
        self._weights = {str(w): w for w in [w_0, w_1, w_2, w_3, w_4, w_5]}
        # print('The solution is:\n   '+'\n   '.join([f'{s} = {self._solution[s]}' for s in [w_0, w_1, w_2, w_3, w_4, w_5]]))

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
        skip: int = 0,
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

        if self.end_frame < 0:
            end = len(self.path) + self.end_frame - self.skip
        else:
            end = self.end_frame - self.skip

        self._start, self._end = start, end

        return (
            self.path[var][start],  # x_0
            self.path[var][end],  # x_1
            self.path.velocity[start][var],  # v_0
            self.path.velocity[end][var],  # v_1
            -self.path.acceleration[var][start],  # a_0
            -self.path.acceleration[var][end],  # a_1
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
        T = np.linspace(0, 1, self._end - self._start)

        x = np.array([self.fits["x"](t) for t in T])
        y = np.array([self.fits["y"](t) for t in T])

        # create an array to match time stamps in simulation to data
        time = np.arange(self._start, self._end + 1)

        return Path(x, y), time

    def __draw__(self):
        trajectory, time = self.simulate()

        f = plt.figure(figsize=(16, 8))
        axes = f.subplot_mosaic(
            """
                AABBDD
                AACCEE
            """
        )

        draw.Tracking(self.path.x, self.path.y, ax=axes["A"])
        draw.Tracking(trajectory.x, trajectory.y, ax=axes["A"], color="salmon")
        axes["A"].scatter(
            [self.fits["x"].x_0, self.fits["x"].x_1],
            [self.fits["y"].x_0, self.fits["y"].x_1],
            color="salmon",
            s=100,
            zorder=100,
        )

        axes["B"].plot(self.path.velocity.x, color=blue_grey)
        axes["B"].plot(time[1:], trajectory.velocity.x, color="salmon")
        axes["B"].scatter(
            [time[0], time[-1]],
            [self.fits["x"].v_0, self.fits["x"].v_1],
            color="salmon",
            s=100,
            zorder=100,
        )

        axes["D"].plot(self.path.velocity.y, color=blue_grey)
        axes["D"].plot(time[1:], trajectory.velocity.y, color="salmon")
        axes["D"].scatter(
            [time[0], time[-1]],
            [self.fits["y"].v_0, self.fits["y"].v_1],
            color="salmon",
            s=100,
            zorder=100,
        )

        axes["C"].plot(self.path.acceleration.x, color=blue_grey)
        axes["C"].plot(time[1:], trajectory.acceleration.x, color="salmon")
        axes["C"].scatter(
            [time[0], time[-1]],
            [self.fits["x"].a_0, self.fits["x"].a_1],
            color="salmon",
            s=100,
            zorder=100,
        )

        axes["E"].plot(self.path.acceleration.y, color=blue_grey)
        axes["E"].plot(time[1:], trajectory.acceleration.y, color="salmon")
        axes["E"].scatter(
            [time[0], time[-1]],
            [self.fits["y"].a_0, self.fits["y"].a_1],
            color="salmon",
            s=100,
            zorder=100,
        )

        axes["B"].set(title="velocity x")
        axes["D"].set(title="velocity y")
        axes["C"].set(title="acceleration x")
        axes["E"].set(title="acceleration y")


if __name__ == "__main__":
    from data import paths
    import draw

    # # load a locomotion bout
    ROI = "T3"
    _bouts = pd.read_hdf(
        paths.analysis_folder
        / "behavior"
        / "saved_data"
        / f"{ROI}_crossings.h5"
    ).sort_values("duration")
    bout = LocomotionBout(_bouts.iloc[2])

    # fit
    msd = MSD(bout, start_frame=2, end_frame=40)
    trajectory, time = msd.simulate()

    # plot results
    msd.__draw__()

    plt.show()
