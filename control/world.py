import numpy as np
from fcutils.maths.geometry import calc_distance_between_points_2d

from control import config
from ._world import simulated, from_tracking


class World:
    curr_traj_waypoint_idx = 1
    moved_to_next = []
    itern = 0

    def __init__(self, trials_cache, trialn):
        if config.TRAJECTORY_CONFIG["traj_type"] == "tracking":
            self.trajectory, self.duration, self.trial = from_tracking(
                trials_cache, trialn
            )
        else:
            self.trajectory, self.duration, self.trial = simulated()

    def plan(self, curr_x):
        """
            Given the current state and the goal trajectory, 
            find the next N sates, based on planning
        """
        traj_length = len(self.trajectory)
        n_ahead = config.PLANNING_CONFIG["n_ahead"]
        pred_len = config.PLANNING_CONFIG["prediction_length"] + 1

        # get the closest traj point in the next chunk of trajectory
        curr_idx = self.curr_traj_waypoint_idx
        min_idx = (
            np.argmin(
                np.linalg.norm(
                    curr_x[:2]
                    - self.trajectory[curr_idx - 1 : curr_idx + 10, :2],
                    axis=1,
                )
            )
            + curr_idx
        )

        # keep track of where in the trajectory we are
        if min_idx + n_ahead != self.curr_traj_waypoint_idx:
            self.moved_to_next.append(self.itern)

        self.curr_traj_waypoint_idx = min_idx
        self.current_traj_waypoint = self.trajectory[min_idx, :]

        start = min_idx + n_ahead  # don't overshoot
        if start > traj_length:
            start = traj_length

        end = min_idx + n_ahead + pred_len

        if start + pred_len > traj_length or start == end:
            return None  # finished!

        # Make sure planned trajectory has the correct length
        if (end - start) != pred_len:
            planned = self.trajectory[start:end]
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
            planned = self.trajectory[start:end]

        if len(planned) != pred_len:
            raise ValueError(
                f"Planned trajecotry length should be {pred_len} but it is {len(planned)} instead"
            )
        else:
            return planned

    def isdone(self, curr_x):
        """
            Checks if the task is complited by seeing if the mouse
            is close enough to the end of the trajectory
        """
        mouse_xy = np.array([curr_x.x, curr_x.y])
        goal_xy = self.trajectory[-1, :2]
        dist = calc_distance_between_points_2d(mouse_xy, goal_xy)

        if dist <= config.TRAJECTORY_CONFIG["min_dist"]:
            return True
        else:
            return False
