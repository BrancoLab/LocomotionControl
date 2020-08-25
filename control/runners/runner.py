from logging import getLogger
from tqdm import tqdm
import numpy as np


class ExpRunner():
    """ experiment runner
    """
    def __init__(self, config):
        """
        """
        self.n_iters = config.TASK_HORIZON

    def run(self, env, controller, planner):
        """
        Returns:
            history_x (numpy.ndarray): history of the state,
            shape(episode length, state_size)
            history_u (numpy.ndarray): history of the state,
            shape(episode length, input_size)
        """
        done = False
        curr_x, info = env.reset()
        history_x, history_u, history_g = [], [], []
        step_count = 0
        score = 0.

        for iter in tqdm(range(self.n_iters)):
            # plan
            g_xs = planner.plan(curr_x, info["goal_state"])

            # obtain sol
            u = controller.obtain_sol(curr_x, g_xs)

            # step
            next_x, cost, done, info = env.step(u)

            # save
            history_u.append(u)
            history_x.append(curr_x)
            history_g.append(g_xs[0])

            # update
            curr_x = next_x
            score += cost
            step_count += 1

            if done:
                break

        print("Controller type = {}, Score = {}"\
                     .format(controller, score))
        return np.array(history_x), np.array(history_u), np.array(history_g), info