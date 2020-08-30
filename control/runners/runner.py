from logging import getLogger
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


class ExpRunner():
    """ experiment runner
    """
    def __init__(self, config, interactive_plot_fn):
        """
        """
        self.n_iters = config.TASK_HORIZON
        self.state = config._state
        self.mouse = config.mouse
        self.params = config.params
        self.interactive_plot = interactive_plot_fn

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


        plt.ion()
        f, axarr = plt.subplots(figsize=(12, 8), ncols=2, nrows=2)
        axarr = axarr.flatten()

        for itern in tqdm(range(self.n_iters)):
            print(f'iteration: {itern}')
            
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

            # update plot
            x = self.state(*next_x)
            goal = self.state(g_xs[0, 0], g_xs[0, 1], g_xs[0, 2])

            self.interactive_plot(axarr, x, goal, u, info, g_xs, itern, self.mouse, self.params, history_u)

            f.canvas.draw()
            plt.pause(0.01)


            if done:
                break
        
        plt.ioff()
        return np.array(history_x), np.array(history_u), np.array(history_g), info