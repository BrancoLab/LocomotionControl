from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

from fcutils.maths.geometry import calc_distance_between_points_2d
from fcutils.plotting.utils import clean_axes
from fcutils.plotting.colors import desaturate_color


from proj.utils import polar_to_cartesian, seagreen, salmon

_xy = namedtuple("xy", "x, y")
_xyt = namedtuple("xyt", "x, y, t")


class World:
    """
        Class to keep a representation of the world (model + trajectory)
        in euclidean representation, regardless of the model's own coordinates system
    """

    def __init__(self, model):
        self.model = model

    # -------------------------------- Initialize -------------------------------- #

    def _initialize_world(self, trajectory):
        # Get the world size from Config
        maxd = self.model.trajectory["distance"]
        world_size = _xy([-maxd, maxd], [-maxd, maxd])

        # check that the trajectory fits in the world size
        if np.min(trajectory[:, 0]) < world_size.x[0]:
            world_size.x[0] = np.min(trajectory[:, 0])

        if np.max(trajectory[:, 0]) > world_size.x[1]:
            world_size.x[1] = np.max(trajectory[:, 0])

        if np.min(trajectory[:, 1]) < world_size.y[0]:
            world_size.y[0] = np.min(trajectory[:, 1])

        if np.max(trajectory[:, 1]) > world_size.y[1]:
            world_size.y[1] = np.max(trajectory[:, 1])

        return world_size

    def _spawn_model_random(self, trajectory, v=0, omega=0):
        # start as a point in the plane with theta 0
        x = np.random.randint(self.world_size.x[0], self.world_size.x[1] / 5)
        y = np.random.randint(self.world_size.y[0], self.world_size.y[1] / 5)

        traj_start = _xy(trajectory[0, 0], trajectory[0, 1])

        if self.model.MODEL_TYPE == "cartesian":
            raise NotImplementedError(
                "Compute theta given position and trajectory"
            )

            # keep track of model's position
            # self.model_position_world = _xyt(x, y, t)

        elif self.model.MODEL_TYPE == "polar":
            # compute r and gamma
            r = calc_distance_between_points_2d(
                [x, y], [traj_start.x, traj_start.y]
            )
            gamma = np.arctan2(traj_start.x - x, traj_start.y - y)

            # update model state
            self.model.curr_x = self.model._state(r, gamma, v, omega)

            # keep track of model's position
            self.model_position_world = _xy(x, y)

    def _spawn_model_trajectory(self, trajectory, v=0, omega=0):
        """
            Spawn model at start/end of trajectory
        """
        if self.model.MODEL_TYPE == "cartesian":
            self.model.curr_x = self.model._state(
                trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], v, omega
            )

            # keep track of model's position
            self.model_position_world = _xyt(
                trajectory[0, 0], trajectory[0, 1], trajectory[0, 2]
            )

        elif self.model.MODEL_TYPE == "polar":
            self.model.curr_x = self.model._state(
                trajectory[-1, 0], trajectory[-1, 1], v, omega
            )

            # keep track of model's position
            self.model_position_world = _xy(
                *polar_to_cartesian(trajectory[-1, 0], trajectory[-1, 1])
            )
            polar_to_cartesian

    def _reset_world_history(self, trajectory):
        # trajectory
        self.initial_trajectory = trajectory

        # model position world
        self.initial_model_position_world = self.model_position_world
        self.model_position_history_world = [self.initial_model_position_world]

    def initialize_world(self, trajectory):
        """
            Create the world and the model at some location
        """
        # create world
        self.world_size = self._initialize_world(trajectory)

        # spawn model
        if self.model.SPAWN_TYPE == "random":
            self._spawn_model_random(trajectory)
        else:
            self._spawn_model_trajectory(trajectory)

        # reset history
        self._reset_world_history(trajectory)

        # create a figure for live plotting
        plt.ion()
        self.f, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.axis("equal")
        clean_axes(self.f)

    # ---------------------------------- Update ---------------------------------- #

    def update_world(self, curr_goals):
        # Get model's position
        if self.model.MODEL_TYPE == "cartesian":
            x, y = self.model.curr_x.x, self.model.curr_x.y
            t = self.model.curr_x.theta

            self.model_position_history_world.append(_xyt(x, y, t))

        else:
            print("polar to cartesian is incorrect in world history")
            x, y = polar_to_cartesian(
                self.model.curr_x.r, self.model.curr_x.gamma
            )

            self.model_position_history_world.append(_xy(x, y))

        self.model_position_world = self.model_position_history_world[-1]

        # Update plots
        self.visualize_world_live(curr_goals)

    # ------------------------------- Live plotting ------------------------------ #

    def visualize_world_live(self, curr_goals):
        ax = self.ax
        ax.clear()

        # plot trajectory
        ax.scatter(
            self.initial_trajectory[::7, 0],
            self.initial_trajectory[::7, 1],
            s=50,
            color=[0.4, 0.4, 0.4],
            lw=1,
            edgecolors="white",
        )

        # plot currently selected goals
        if self.model.MODEL_TYPE == "cartesian":
            ax.plot(
                curr_goals[:, 0],
                curr_goals[:, 1],
                lw=10,
                color=salmon,
                zorder=-1,
                solid_capstyle="round",
            )
        else:
            raise NotImplementedError("current goals for polar")

        # plot position history
        X = [pos.x for pos in self.model_position_history_world]
        Y = [pos.y for pos in self.model_position_history_world]

        ax.plot(
            X,
            Y,
            lw=9,
            color=desaturate_color(seagreen),
            zorder=-1,
            solid_capstyle="round",
        )

        # plot current position
        x, y = self.model_position_world.x, self.model_position_world.y

        ax.scatter(
            x, y, s=350, color=seagreen, lw=1.5, edgecolors=[0.3, 0.3, 0.3]
        )

        if self.model.MODEL_TYPE == "cartesian":
            # plot body axis
            t = self.model_position_world.t
            dx = np.cos(t) * self.model.mouse["length"]
            dy = np.sin(t) * self.model.mouse["length"]

            ax.plot([x, x + dx], [y, y + dy], lw=8, color=seagreen)
            ax.scatter(
                x + dx,
                y + dy,
                s=225,
                color=seagreen,
                lw=1.5,
                edgecolors=[0.3, 0.3, 0.3],
            )

        # display plot
        self.f.canvas.draw()
        plt.pause(0.001)
