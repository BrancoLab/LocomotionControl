import numpy as np
from fcutils.maths.geometry import calc_distance_between_points_2d
import shutil

from proj.environment.trajectories import (
    parabola,
    sin,
    circle,
    line,
    point,
    from_tracking,
)
from proj.utils import traj_to_polar, timestamp
from proj.environment.world import World
from proj.paths import (
    frames_cache,
    winstor_trial_cache,
    trials_cache,
    winstor_main,
    main_fld,
)


class Environment(World):
    itern = 0
    moved_to_next = []
    curr_traj_waypoint_idx = None

    traj_funcs = dict(
        parabola=parabola,
        sin=sin,
        circle=circle,
        line=line,
        point=point,
        tracking=from_tracking,
    )

    def __init__(self, model, winstor=False):
        World.__init__(self, model)

        self.model = model

        try:
            self._traj_func = self.traj_funcs[model.trajectory["name"]]
        except KeyError:
            raise ValueError(
                f'Could not find a trajectory constructing curve called: {model.trajectory["name"]}'
            )

        # Prepare paths
        self.winstor = winstor
        if not winstor:
            self.main_fld = main_fld
            self.cache_fld = frames_cache
            self.trials_cache = trials_cache
        else:
            self.main_fld = winstor_main
            self.cache_fld = (
                winstor_main / f'{model.trajectory["name"]}_{timestamp()}'
            )
            self.trials_cache = winstor_trial_cache

    def make_trajectory(self):
        """
            Defines a path that the mouse has to 
            follow, given a dictionary of params.
            Path is shaped as a parabola
        """
        params = self.model.trajectory
        n_steps = int(params["nsteps"])

        traj = self._traj_func(
            n_steps, params, self.model.planning, self.trials_cache
        )

        if self.model.MODEL_TYPE == "polar":
            traj = traj_to_polar(traj)

        return traj

    def reset(self):
        """
            resets stuff
        """
        # reset model
        self.model.reset()

        # make goal trajetory
        trajectory, goal_duration = self.make_trajectory()

        self.goal_duration = goal_duration  # how long it should take

        # reset the world and the model's initial position
        self.initialize_world(trajectory)

        # empty cache frames folder
        try:
            shutil.rmtree(str(frames_cache))
        except (FileNotFoundError, PermissionError):
            pass
        frames_cache.mkdir(exist_ok=True)

        return trajectory

    def plan(self, curr_x, g_traj, itern):
        """
            Given the current state and the goal trajectory, 
            find the next N sates, based on planning
        """
        n_ahead = self.model.planning["n_ahead"]
        pred_len = self.model.planning["prediction_length"] + 1

        min_idx = np.argmin(np.linalg.norm(curr_x[:2] - g_traj[:, :2], axis=1))

        # keep track of where in the trajectory we are
        if min_idx + n_ahead != self.curr_traj_waypoint_idx:
            self.moved_to_next.append(self.itern)

        self.curr_traj_waypoint_idx = min_idx + n_ahead
        self.current_traj_waypoint = g_traj[min_idx, :]

        start = min_idx + n_ahead
        if start > len(g_traj):
            start = len(g_traj)

        end = min_idx + n_ahead + pred_len

        if (min_idx + n_ahead + pred_len) > len(g_traj):
            end = len(g_traj)

        if abs(start - end) != pred_len:
            return np.tile(g_traj[-1], (pred_len, 1))

        return g_traj[start:end]

    def isdone(self, curr_x, trajectory):
        """
            Checks if the task is complited by seeing if the mouse
            is close enough to the end of the trajectory
        """
        if self.model.MODEL_TYPE == "cartesian":
            mouse_xy = np.array([curr_x.x, curr_x.y])
            goal_xy = trajectory[-1, :2]

            dist = calc_distance_between_points_2d(mouse_xy, goal_xy)
        else:
            dist = curr_x.r

        if dist <= self.model.trajectory["min_dist"]:
            return True
        else:
            return False
