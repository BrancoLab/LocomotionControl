import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from collections import namedtuple

from PythonLinearNonlinearControl.envs.env import Env
from PythonLinearNonlinearControl.plotters.plot_objs import circle_with_angle, square, circle

state = namedtuple('state', 'x, y, theta, v')
control = namedtuple('control', 'L, R')
m = 10

def step_two_wheeled_env(curr_x, u, dt, method="Oylar"):
    """ step two wheeled enviroment
    
    Args:
        curr_x (numpy.ndarray): current state, shape(state_size, )
        u (numpy.ndarray): input, shape(input_size, )
        dt (float): sampling time
    Returns:
        next_x (numpy.ndarray): next state, shape(state_size. )
    
    """
    u = control(*u)
    x = state(*curr_x)

    dxdt = np.array([
        x.v * np.cos(x.theta),
        x.v * np.sin(x.theta),
        (u.R - u.L) / m,
        (u.R + u.L)/m + (1 - (u.R - u.L)/(u.R + u.L))
    ])

    next_x = dxdt.flatten() * dt + curr_x

    return next_x


class AlloEnv(Env):
    """ Two wheeled robot with constant goal Env
    """
    def __init__(self):
        """
        """
        self.config = {"state_size" : 4,
                       "input_size" : 2,
                       "dt" : 0.01,
                       "max_step" : 1000,
                       "input_lower_bound": (0, 0),
                       "input_upper_bound": (2, 2),
                       }

        super(AlloEnv, self).__init__(self.config)
    
    @staticmethod
    def make_road(linelength=3., circle_radius=1.):
        """ make track
        Returns:
            road (numpy.ndarray): road info, shape(n_point, 3) x, y, angle
        """
        # line
        # not include start points
        line = np.linspace(-1.5, 1.5, num=51, endpoint=False)[1:]
        line_1 = np.stack((line, np.zeros(50)), axis=1)
        line_2 = np.stack((line[::-1], np.zeros(50)+circle_radius*2.), axis=1)

        # circle
        circle_1_x, circle_1_y = circle(linelength/2., circle_radius,
            circle_radius, start=-np.pi/2., end=np.pi/2., n_point=50)
        circle_1 = np.stack((circle_1_x , circle_1_y), axis=1)
        
        circle_2_x, circle_2_y = circle(-linelength/2., circle_radius,
            circle_radius, start=np.pi/2., end=3*np.pi/2., n_point=50)
        circle_2 = np.stack((circle_2_x , circle_2_y), axis=1)

        road_pos = np.concatenate((line_1, circle_1, line_2, circle_2), axis=0)

        # calc road angle
        road_diff = road_pos[1:] - road_pos[:-1]
        road_angle = np.arctan2(road_diff[:, 1], road_diff[:, 0]) 
        road_angle = np.concatenate((np.zeros(1), road_angle))
        road_vel = np.ones_like(road_angle) * 50

        road = np.concatenate((road_pos, road_angle[:, np.newaxis], road_vel[:, np.newaxis]), axis=1)
        road =  np.tile(road, (3, 1)) 

        # plt.plot(road[:, 0], road[:, 1])
        # plt.show()

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
        next_x = step_two_wheeled_env(self.curr_x, u, self.config["dt"])

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
