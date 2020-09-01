from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from proj.plotting.live import update_interactive_plot

def run_experiment(environment, controller, model, n_steps=200):
    """
        Runs an experiment

        :param environment: instance of Environment, is used to specify 
            a goals trajectory (reset) and to identify the next goal 
            states to be considered (plan)

        :param controller: isntance of Controller, used to compute controls

        :param model: instance of Model

        :param n_steps: int, number of steps in iteration

        :returns: the history of events as stored by model
    """
    # reset things
    model.reset()
    trajectory = environment.reset()

    # setup interactive plot
    plt.ion()
    f, axarr = plt.subplots(figsize=(12, 8), ncols=2, nrows=2)
    axarr = axarr.flatten()

    # RUN
    for itern in tqdm(range(n_steps)):
        curr_x = np.array(model.curr_x)
        # plan
        g_xs = environment.plan(curr_x, trajectory)

        # obtain sol
        u = controller.obtain_sol(curr_x, g_xs)

        # step
        model.step(u)

        # update interactieve plot
        goal= model._state(g_xs[0, 0], g_xs[0, 1], g_xs[0, 2], g_xs[0, 3], g_xs[0, 4])
        update_interactive_plot(axarr, model, goal, trajectory, g_xs, itern)


        f.canvas.draw()
        plt.pause(0.01)

    return model.history