from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from proj.plotting.live import (
    update_interactive_plot_manual,
    update_interactive_plot_manual_polar,
)


def make_figure(model_type):
    if model_type == "cartesian":
        f, ax = plt.subplots(figsize=(16, 8))
        return f, ax
    else:
        f = plt.figure(figsize=(22, 8))
        ax = f.add_subplot(121)
        pax = f.add_subplot(122, projection="polar")

        return f, [ax, pax]


def run_manual(
    environment,
    model,
    n_secs,
    u,
    traj=None,
    plot=True,
    folder=None,
    ax_kwargs={},
):
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
        f, ax = make_figure(model.MODEL_TYPE)

    model.move_to_random_location()

    # get number of steps
    n_steps = int(n_secs / model.dt)

    # RUN
    for itern in tqdm(range(n_steps)):
        # curr_x = np.array(model.curr_x)

        # step
        model.step(u[itern, :])

        # update interactieve plot
        if plot:
            if model.MODEL_TYPE == "cartesian":
                update_interactive_plot_manual(ax, model)
            else:
                update_interactive_plot_manual_polar(model, *ax, traj=traj)

            f.canvas.draw()

            if not isinstance(ax, list):
                ax.set(**ax_kwargs)
                ax.axis("off")

            plt.pause(0.01)

            if folder is not None:
                if itern < 10:
                    n = f"0{itern}"
                else:
                    n = str(itern)
                f.savefig(str(folder / n))

    # SAVE results
    # model.save()

    return folder
