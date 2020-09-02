from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from proj.plotting.live import update_interactive_plot

def run_experiment(environment, controller, model, n_steps=200, plot=True, folder=None):
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
    if folder is not None:
        model.save_folder = Path(folder)
        
    # reset things
    model.reset()
    trajectory = environment.reset()

    # setup interactive plot
    if plot:
        plt.ion()
        f, axarr = plt.subplots(figsize=(16, 8), ncols=3, nrows=2)
        axarr = axarr.flatten()

    # RUN
    for itern in tqdm(range(n_steps)):
        curr_x = np.array(model.curr_x)
        # plan
        g_xs = environment.plan(curr_x, trajectory, itern)

        # obtain sol
        u = controller.obtain_sol(curr_x, g_xs)

        # step
        model.step(u)

        # Check if we're done
        if environment.isdone(model.curr_x, trajectory):
            print(f'Reached end of trajectory after {itern} steps')
            break

        # update interactieve plot
        if plot:
            goal= model._state(g_xs[0, 0], g_xs[0, 1], g_xs[0, 2], g_xs[0, 3], g_xs[0, 4])
            update_interactive_plot(axarr, model, goal, trajectory, g_xs, itern)


            f.canvas.draw()
            plt.pause(0.01)

    # SAVE results
    model.save(trajectory)

    return model.history