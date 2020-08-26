from logging import getLogger
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


class ExpRunner():
    """ experiment runner
    """
    def __init__(self, config):
        """
        """
        self.n_iters = config.TASK_HORIZON
        self.state = config._state

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
        f, axarr = plt.subplots(figsize=(6, 4), ncols=2)

        for iter in tqdm(range(self.n_iters)):
            # plan
            g_xs = planner.plan(curr_x, info["goal_state"])

            # print(f'\nGoal: {[round(p, 2) for p in g_xs[0, :]]}')

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
            goal = self.state(g_xs[0, 0], g_xs[0, 1], g_xs[0, 2], g_xs[0, 3])

            axarr[0].clear()
            axarr[1].clear()
            axarr[0].scatter(info['goal_state'][:, 0], info['goal_state'][:, 1], 
                        color='g', alpha=.2, zorder=-1)

            axarr[0].plot(g_xs[:, 0], g_xs[:, 1], 
                        lw=3, color='k', alpha=1, zorder=-1)

            axarr[0].scatter(x.x, x.y, s=160, c=x.v, vmin=-10, vmax=10, cmap='bwr', 
                                    lw=2, edgecolors='k')
            # wedge = Wedge((x.x, x.y), .4, theta1=np.degrees(x.theta) - 15,
            #                 theta2=np.degrees(x.theta) + 15, width=.2, color='red')
            # axarr[0].add_patch(wedge)
            axarr[0].plot([x.x, x.x + np.cos(x.theta) + .15],
                            [x.y, x.y + np.sin(x.theta) + .15],
                            lw=2, color='r', zorder=99)
            axarr[0].set(title=f'ITER: {iter} | x:{round(x.x, 2)}, y:{round(x.y, 2)}, ' +
                                f' theta:{round(np.degrees(x.theta), 2)}, v:{round(x.v, 2)}\n'+
                                f'GOAL: x:{round(goal.x, 2)}, y:{round(goal.y, 2)}, ' +
                                f' theta:{round(np.degrees(goal.theta), 2)}, v:{round(goal.v, 2)}'
                                )

            axarr[1].bar([0, 1], u, color=['b', 'r'])
            axarr[1].set(title='control', xticks=[0, 1], xticklabels=['L', 'R'])

            f.canvas.draw()
            plt.pause(0.01)


            if done:
                break
        
        plt.ioff()
        print("Controller type = {}, Score = {}"\
                     .format(controller, score))
        return np.array(history_x), np.array(history_u), np.array(history_g), info