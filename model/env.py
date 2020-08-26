import numpy as np

from control.envs.env import Env
from control.plotters.plot_objs import circle_with_angle, square, circle

from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d

class AlloEnv(Env):
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

        super(AlloEnv, self).__init__(self.config)


    @staticmethod
    def make_road(linelength=1., circle_radius=1.):
        """ make track
        Returns:
            road (numpy.ndarray): road info, shape(n_point, 3) x, y, angle
        """
        # not include start points
        line = np.linspace(-linelength/2, linelength/2, 
                                num=11, endpoint=False)[1:]
        line_1 = np.stack((line, np.zeros(10)), axis=1)
        line_2 = np.stack((line[::-1], np.zeros(10)+circle_radius*2.), axis=1)

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
        road_vel = np.ones_like(road_angle)  + 1


        road = np.concatenate((road_pos, road_angle[:, np.newaxis], road_vel[:, np.newaxis]), axis=1)
        road =  np.tile(road, (3, 1))

        # x = np.linspace(0, 11, 101)
        # y = np.sin(x*5)
        # angle = np.radians(calc_angle_between_points_of_vector_2d(x, y))
        # v = np.ones_like(x) * 5

        # road = np.vstack([x, y, angle, v]).T

        # ? plot road
        # import matplotlib.pyplot as plt
        # # plt.scatter(road[:, 0], road[:, 1], c=road[:, 2], vmin=0, vmax=np.pi, cmap='bwr')
        # plt.scatter(x, y, c=np.degrees(angle), vmin=0, vmax=360, cmap='bwr')
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
