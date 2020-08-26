import numpy as np

from measurement.measures import Distance, Time


from control.envs.env import Env
from control.plotters.plot_objs import circle_with_angle, square, circle

from models.allo_two_wheels.utils import make_road

from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d

class Environment(Env):
    def __init__(self, config, model):
        self.config = {"state_size" : config.STATE_SIZE,
                       "input_size" : config.INPUT_SIZE,
                       "dt" : config.DT,
                       "max_step" : config.TASK_HORIZON,
                       "input_lower_bound" : config.INPUT_LOWER_BOUND,
                       "input_upper_bound" : config.INPUT_UPPER_BOUND,
                       }
        self.m = config.m
        self.params = config.params
        self.model = model

        super(Environment, self).__init__(self.config)

    def reset(self, init_x=None):
        """ reset state
        Returns:
            init_x (numpy.ndarray): initial state, shape(state_size, )  
            info (dict): information
        """
        self.step_count = 0
        
        # goal
        self.g_traj = make_road(self.params)

        self.curr_x = self.g_traj[0, :]

        if init_x is not None:
            self.curr_x = init_x

        
        # clear memory
        self.history_x = []
        self.history_g_x = []

        return self.curr_x, {"goal_state": self.g_traj}

    def _compute_step(self, curr_x, u, dt):
        """ step two wheeled enviroment
        
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            u (numpy.ndarray): input, shape(input_size, )
            dt (float): sampling time
        Returns:
            next_x (numpy.ndarray): next state, shape(state_size. )
        
        """
        return self.model.predict_next_state(curr_x, u)

    def step(self, u):
        """ step environments
        Args:
            u (numpy.ndarray) : input, shape(input_size, )
        Returns:
            next_x (numpy.ndarray): next state, shape(state_size, ) 
            cost (float): costs
            done (bool): end the simulation or not
            info (dict): information 
        """
        # clip action
        u = np.clip(u,
                    self.config["input_lower_bound"],
                    self.config["input_upper_bound"])

        # step
        next_x = self._compute_step(self.curr_x, u, self.config["dt"])

        costs = 0.
        costs += 0.1 * np.sum(u**2)
        costs += np.min(np.linalg.norm(self.curr_x - self.g_traj, axis=1))

        # save history
        self.history_x.append(next_x.flatten())
        
        # update
        self.curr_x = next_x.flatten()
        
        # update costs
        self.step_count += 1

        return next_x.flatten(), costs, \
               self.step_count > self.config["max_step"], \
               {"goal_state" : self.g_traj}
