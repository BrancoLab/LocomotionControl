import numpy as np

from control.envs.env import Env
from control.plotters.plot_objs import circle_with_angle, square, circle

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
        self.model = model

        super(Environment, self).__init__(self.config)


    @staticmethod
    def make_road(linelength=1., circle_radius=1.):
        """ make track
        Returns:
            road (numpy.ndarray): road info, shape(n_point, 3) x, y, angle
        """
        time        = np.arange(0, 10, 0.1)
        x = np.linspace(0, 11*3, len(time))
        y = np.sin(time)
        angle = np.radians(calc_angle_between_points_of_vector_2d(x, y))
        v = np.ones_like(x) + 5
        road = np.vstack([x, y, angle, v]).T
        road = np.tile(road, (3, 1))

        # ? plot road
        import matplotlib.pyplot as plt
        # plt.scatter(road[:, 0], road[:, 1], c=road[:, 2], vmin=0, vmax=np.pi, cmap='bwr')
        plt.scatter(x, y, c=np.degrees(angle), vmin=0, vmax=360, cmap='bwr')
        plt.show()

        return road

    def reset(self, init_x=None):
        """ reset state
        Returns:
            init_x (numpy.ndarray): initial state, shape(state_size, )  
            info (dict): information
        """
        self.step_count = 0
        
        self.curr_x = np.zeros(self.config["state_size"])

        if init_x is not None:
            self.curr_x = init_x

        # goal
        self.g_traj = self.make_road()
        
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
