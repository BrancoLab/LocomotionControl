import numpy as np
from scipy.optimize import curve_fit
from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d

class Environment():
    def __init__(self, model):
        self.model = model

    @staticmethod
    def curve(x, a, b, c):
        return (a * (x-b)**2) + + c

    def make_trajectory(self):
        """
            Defines a path that the mouse has to 
            follow, given a dictionary of params.
            Path is shaped as a parabola
        """
        params = self.model.trajectory
        n_steps = int(params['nsteps'])

        # Define 3 points
        X = [0, params['distance']/2, params['distance']]
        Y = [0, params['distance']/4, 0]

        # fit curve and make trace
        coef,_ = curve_fit(self.curve, X, Y)

        x = np.linspace(0, params['distance'], n_steps)
        y = self.curve(x, *coef)

        # Compute other variables that figure in the state vector
        angle = np.radians(calc_angle_between_points_of_vector_2d(x, y))

        speed = (1 - np.sin(np.linspace(0, 3, len(x)))) 
        speed = speed * (params['max_speed']-params['min_speed']) + params['min_speed']

        ang_speed = np.ones_like(speed) # it will be ignored


        trajectory = np.vstack([x, y, angle, speed, ang_speed]).T
        return trajectory[1:, :]

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