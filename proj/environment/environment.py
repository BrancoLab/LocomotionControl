import numpy as np
from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d, calc_distance_between_points_2d

from proj.environment.trajectories import parabola

class Environment():
    traj_funcs = dict(
        parabola = parabola,
    )
    def __init__(self, model):
        self.model = model

        try:
            self._traj_func = self.traj_funcs[model.trajectory['name']]
        except KeyError as e:
            raise ValueError(f'Could not find a trajectory constructing curve called: {model.trajectory["name"]}')

    def make_trajectory(self):
        """
            Defines a path that the mouse has to 
            follow, given a dictionary of params.
            Path is shaped as a parabola
        """
        params = self.model.trajectory
        n_steps = int(params['nsteps'])

        return self._traj_func(n_steps, params)

    def reset(self):
        """
            resets stuff
        """
        # reset model
        self.model.reset()

        # make goal trajetory
        g_traj = self.make_trajectory()

        # Set model's state to the start of the trajectory
        self.model.curr_x = self.model._state(g_traj[0, 0], g_traj[0, 1], g_traj[0, 2], 0, 0)
        

        return g_traj

    def plan(self, curr_x, g_traj):
        """
            Given the current state and the goal trajectory, 
            find the next N sates, based on planning
        """
        n_ahead = self.model.planning['n_ahead']
        pred_len = self.model.planning['prediction_length'] + 1

        min_idx = np.argmin(np.linalg.norm(curr_x[:2] - g_traj[:, :2],
                                    axis=1))

        start = (min_idx + n_ahead) 
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
        mouse_xy = np.array([curr_x.x, curr_x.y])
        goal_xy = trajectory[-1, :2]

        dist = calc_distance_between_points_2d(mouse_xy, goal_xy)

        if dist <= self.model.trajectory['min_dist']:
            return True
        else:
            return False