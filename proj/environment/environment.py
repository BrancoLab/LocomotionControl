import numpy as np
import logging

from fcutils.maths.geometry import calc_distance_between_points_2d

from proj.environment.trajectories import (
    parabola,
    sin,
    circle,
    line,
    point,
    from_tracking,
    simulated_but_realistic,
)
from proj.environment.world import World
from proj.environment.manager import Manager


class Environment(World, Manager):
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
        real_simulated=simulated_but_realistic,
    )

    def __init__(self, model, winstor=False):
        World.__init__(self, model)
        Manager.__init__(self, model, winstor=winstor)

        self.model = model

        try:
            self._traj_func = self.traj_funcs[model.trajectory["name"]]
        except KeyError:
            raise ValueError(
                f'Could not find a trajectory constructing curve called: {model.trajectory["name"]}'
            )

        # Prepare paths
        self.winstor = winstor

    def make_trajectory(self):
        """
            Defines a path that the mouse has to 
            follow, given a dictionary of params.
            Path is shaped as a parabola
        """
        traj = self._traj_func(
            self.model.trajectory, self.model.planning, self.trials_cache
        )

        return traj

    def reset(self):
        """
            resets stuff
        """
        # reset model
        self.model.reset()

        # make goal trajetory
        (trajectory, goal_duration), trial = self.make_trajectory()
        self.trial = trial

        self.goal_duration = goal_duration  # how long it should take

        # reset the world and the model's initial position
        self.initialize_world(trajectory)

        return trajectory

    def plan(self, curr_x, g_traj, itern):
        """
            Given the current state and the goal trajectory, 
            find the next N sates, based on planning
        """

        traj_length = len(g_traj)
        n_ahead = self.model.planning["n_ahead"]
        pred_len = self.model.planning["prediction_length"] + 1

        min_idx = np.argmin(np.linalg.norm(curr_x[:2] - g_traj[:, :2], axis=1))

        # keep track of where in the trajectory we are
        if min_idx + n_ahead != self.curr_traj_waypoint_idx:
            self.moved_to_next.append(self.itern)

        self.curr_traj_waypoint_idx = min_idx + n_ahead
        self.model.curr_traj_waypoint_idx = self.curr_traj_waypoint_idx
        self.current_traj_waypoint = g_traj[min_idx, :]

        start = min_idx + n_ahead  # don't overshoot
        if start > traj_length:
            start = traj_length

        end = min_idx + n_ahead + pred_len

        if start + pred_len > traj_length:
            return None  # finished!
            # end = traj_length

        # Make sure planned trajectory has the correct length
        if start == end:
            return None  # finished!

        if (end - start) != pred_len:
            planned = g_traj[start:end]
            len_diff = len(planned) - pred_len

            if len_diff <= 0:
                try:
                    planned = np.pad(
                        planned, ((0, abs(len_diff)), (0, 0)), mode="edge"
                    )
                except ValueError:
                    raise ValueError(
                        f"Padding went wrong for planned with shape {planned.shape} and len_diff {len_diff}"
                    )

            else:
                raise ValueError("Something went wrong")
        else:
            planned = g_traj[start:end]

        if len(planned) != pred_len:
            raise ValueError(
                f"Planned trajecotry length should be {pred_len} but it is {len(planned)} instead"
            )
        else:
            return planned

    def isdone(self, curr_x, trajectory):
        """
            Checks if the task is complited by seeing if the mouse
            is close enough to the end of the trajectory
        """
        mouse_xy = np.array([curr_x.x, curr_x.y])
        goal_xy = trajectory[-1, :2]
        dist = calc_distance_between_points_2d(mouse_xy, goal_xy)

        if dist <= self.model.trajectory["min_dist"]:
            logging.info(
                f"Reached the end of the threshold as we're within {self.model.trajectory['min_dist']} from the end"
            )
            return True
        else:
            return False
