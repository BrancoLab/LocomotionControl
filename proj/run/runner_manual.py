from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from proj.plotting.live import update_interactive_plot_manual


def run_manual(environment, model, n_steps, u, plot=True, folder=None, ax_kwargs={}):
    """
        Runs an experiment where the controls U have been determined 
        already. U should be a vector with shape n_steps x n_inputs.
    """

    if folder is not None:
        folder = Path(folder)
        model.save_folder = folder

    # setup interactive plot
    if plot:
        plt.ion()
        f, ax = plt.subplots(figsize=(16, 8))

    # RUN
    for itern in tqdm(range(n_steps)):
        curr_x = np.array(model.curr_x)

        # step
        model.step(u[itern, :])

        # update interactieve plot
        if plot:
            update_interactive_plot_manual(ax, model)

            f.canvas.draw()

            ax.set(**ax_kwargs)
            ax.axis('off')

            plt.pause(0.01)

            if folder is not None:
                if itern < 10:
                    n = f'0{itern}'
                else:
                    n = str(itern)
                f.savefig(str(folder / n))

    # SAVE results
    # model.save()

    return folder