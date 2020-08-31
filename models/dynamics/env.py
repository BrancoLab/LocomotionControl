import numpy as np

from control.envs.env import Env
from control.plotters.plot_objs import circle_with_angle, square, circle

from models.dynamics.utils import make_road

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
        self.g_traj = make_road(self.params)[1:, :]

        self.curr_x = self.g_traj[0, :]

        if init_x is not None:
            self.curr_x = init_x

        # clear memory
        self.history_x = []
        self.history_g_x = []

        self.last_dxdt = None
        self.nu = None
        self.model.env_last_dxdt = None
        self.model.env_nu = None

        self.model.reset_ldxdt_nu()

        return self.curr_x, {"goal_state": self.g_traj}

    def _predict_next(self, x, u):
        if self.last_dxdt is None:
            self.last_dxdt = self.model._state(*np.zeros(4))
        if self.nu is None:
            self.nu = self.model._control(*np.zeros(2))

        return self.model.predict_next_state(x, u, last_dxdt = self.last_dxdt, nu = self.nu)

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

        # print(f'Env stepping with control: ', u)

        # step
        next_x, last_dxdt, nu = self._predict_next(self.curr_x, u)
        self.last_dxdt = last_dxdt
        self.nu = nu
        self.model.env_last_dxdt = last_dxdt
        self.model.env_nu = nu

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
