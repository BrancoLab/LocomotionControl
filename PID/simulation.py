import sys

sys.path.append("./")

from loguru import logger
from tpd import recorder
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

recorder.start(base_folder="./logs", name="pid_sim", timestamp=False)

from myterial import salmon, blue_grey_dark
from fcutils.progress import track

from data.dbase.hairpin_trace import HairpinTrace

from PID.state import State
from PID.robot import Robot
from PID.pid import PID


class History:
    x: list = []
    y: list = []
    l_wheel_avel: list = []
    r_wheel_avel: list = []
    angle_error: list = []
    speed_error: list = []
    goal_idx: list = []

    def add(
        self,
        robot: Robot,
        angle_error: float,
        speed_error: float,
        goal_idx: int,
    ):
        self.x.append(robot.state.x)
        self.y.append(robot.state.y)
        self.l_wheel_avel.append(robot.l_wheel_avel)
        self.r_wheel_avel.append(robot.r_wheel_avel)

        self.angle_error.append(angle_error)
        self.speed_error.append(speed_error)
        self.goal_idx.append(goal_idx)


class Simulation:
    def __init__(self, dt: float = 0.001, look_ahead: int = 3):
        self.dt = dt
        self.look_ahead = look_ahead

        # initialize controllers
        angle_pid = PID(
            proportional_gain=1e-5,
            integral_gain=0,
            derivative_gain=1e-5,
            dt=dt,
        )
        speed_pid = PID(1e-2, 0, 1e-4, dt=dt)

        # get track
        self.hp_trace = HairpinTrace()
        self.hp_trace.trace_orientation = np.radians(
            self.hp_trace.trace_orientation
        )
        self.hp_trace.trace_orientation[0] = self.hp_trace.trace_orientation[1]

        # initialize robot
        initial_state = State(
            self.hp_trace.trace[0, 0],
            self.hp_trace.trace[0, 1],
            self.hp_trace.trace_orientation[0],
        )
        self.robot = Robot(initial_state, angle_pid, speed_pid, dt)

        # other variables
        self.history = History()

    def _get_next_goal(self) -> Tuple[int, State]:
        """
            Get the closest point on the trace and look ahad a bit
        """
        point = np.array([self.robot.state.x, self.robot.state.y])
        dist = np.linalg.norm(self.hp_trace.trace - point, axis=1)
        goal_idx = np.argmin(dist) + self.look_ahead

        if goal_idx + self.look_ahead >= len(self.hp_trace.trace):
            return None, None
        else:
            return (
                goal_idx,
                State(
                    self.hp_trace.trace[goal_idx, 0],
                    self.hp_trace.trace[goal_idx, 1],
                    self.hp_trace.trace_orientation[goal_idx],
                ),
            )

    def run(self, duration: float = 3):
        n_steps = int(duration / self.dt)
        logger.info(f"Running simulation for {duration}s ({n_steps} steps)")

        for step in track(
            np.arange(n_steps),
            total=n_steps,
            description="running simulation...",
        ):
            # get goal state
            goal_idx, goal = self._get_next_goal()
            if goal is None:  # reached the end
                break

            # apply robot's control
            angle_error, speed_error = self.robot.control(goal)

            # move robot
            self.robot.move()

            # update history
            self.history.add(self.robot, angle_error, speed_error, goal_idx)

    def visualize(self):
        f = plt.figure(figsize=(16, 8))
        axes = f.subplot_mosaic(
            """
            AAABBB
            AAACCC
            AAADDD
            EEEFFF
            """
        )

        axes["A"].plot(
            self.hp_trace.trace[:, 0],
            self.hp_trace.trace[:, 1],
            lw=2,
            color=blue_grey_dark,
        )
        axes["A"].scatter(
            self.hp_trace.trace[::2, 0],
            self.hp_trace.trace[::2, 1],
            c=self.hp_trace.trace_orientation[::2],
            cmap="bwr",
            vmin=0,
            vmax=6.28,
            s=100,
            zorder=100,
            lw=1,
            ec="k",
        )
        axes["A"].plot(self.history.x, self.history.y, "o-", color=salmon)

        axes["B"].plot(self.history.l_wheel_avel, label="L")
        axes["B"].plot(self.history.r_wheel_avel, label="R")
        axes["B"].legend()

        axes["F"].plot(self.history.goal_idx)

        axes["C"].plot(self.hp_trace.trace_orientation)

        axes["E"].plot(self.history.angle_error, label="angle")
        axes["E"].plot(self.history.speed_error, label="speed")
        axes["E"].legend()

        plt.show()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
    sim.visualize()

    # TODO get it to work
