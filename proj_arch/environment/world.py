from collections import namedtuple
import numpy as np

from proj.environment.plotter import Plotter

_xy = namedtuple("xy", "x, y")
_xyt = namedtuple("xyt", "x, y, t")


class World(Plotter):
    """
        Class to keep a representation of the world (model + trajectory)
        in euclidean representation, regardless of the model's own coordinates system
    """

    total_cost = 0

    stop = False
    _cache = dict(speed_plot_x=[], speed_plot_y=[],)

    def __init__(self, model):
        Plotter.__init__(self)

        self.model = model
        self.plot_every = self.model.traj_plot_every

        keys = list(model._state._fields)
        keys.extend(model._control._fields)
        self.cost_history = {k: [] for k in keys}
        self.cost_history["total"] = []

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

    def _spawn_model_trajectory(self, trajectory, v=0, omega=0):
        """
            Spawn model at start/end of trajectory
        """
        self.model.curr_x = self.model._state(
            trajectory[0, 0],
            trajectory[0, 1],
            trajectory[0, 2],
            trajectory[0, 3],
            trajectory[0, 4],
        )

        # keep track of model's position
        self.model_position_world = _xyt(
            trajectory[0, 0], trajectory[0, 1], trajectory[0, 2]
        )

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
        if trajectory is None:
            return

        # create world
        self.world_size = self._initialize_world(trajectory)

        # spawn model
        self._spawn_model_trajectory(trajectory)

        # reset history
        self._reset_world_history(trajectory)

        # create a figure for live plotting
        self.make_figure()

    # ---------------------------------- Update ---------------------------------- #
    def _update_cost_history(self):
        self.total_cost += self.curr_cost["total"]
        self.cost_history["total"].append(self.curr_cost["total"])

        for k in self.model._state._fields:
            self.cost_history[k].append(self.curr_cost["state"]._asdict()[k])
        for k in self.model._control._fields:
            self.cost_history[k].append(self.curr_cost["control"]._asdict()[k])

    def update_world(self, curr_goals, elapsed=None):
        try:
            self._update_cost_history()
        except Exception:
            pass

        # Get model's position
        x, y = self.model.curr_x.x, self.model.curr_x.y
        t = self.model.curr_x.theta
        self.model_position_history_world.append(_xyt(x, y, t))
        self.model_position_world = self.model_position_history_world[-1]

        # Update plots
        if self.model.LIVE_PLOT:
            self.visualize_world_live(curr_goals, elapsed=elapsed)
