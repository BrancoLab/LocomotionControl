from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
)
import numpy as np
from pathlib import Path

from proj.paths import frames_cache, main_fld
from proj.animation.animate import animate_from_images
from proj.utils import timestamp

# define progress bar
progress = Progress(
    TextColumn(
        "[bold, magenta]Step {task.completed}/{task.total} - {task.speed} steps/s"
    ),
    # "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    "•",
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeRemainingColumn(),
)


# run
def run_experiment(
    environment,
    controller,
    model,
    n_secs=30,
    plot=True,
    folder=None,
    frames_folder=None,
):
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
    trajectory = environment.reset()
    model.reset()

    # Get number of steps
    n_steps = int(n_secs / model.dt)
    print(
        f"Starting simulation with {n_steps} steps [{n_secs} at {model.dt} s/step]"
    )

    # RUN
    with progress:
        task_id = progress.add_task("running", start=True, total=n_steps)

        for itern in range(n_steps):
            progress.advance(task_id, 1)

            curr_x = np.array(model.curr_x)

            # plan
            g_xs = environment.plan(curr_x, trajectory, itern)

            # obtain sol
            u = controller.obtain_sol(curr_x, g_xs)

            # step
            model.step(u)

            # update world
            environment.itern = itern
            environment.update_world(g_xs, elapsed=itern * model.dt)

            # Check if we're done
            if (
                environment.isdone(model.curr_x, trajectory)
                or environment.stop
            ):
                print(f"Reached end of trajectory after {itern} steps")
                break

    # SAVE results
    # model.save(trajectory)

    # make gif
    try:
        animate_from_images(
            str(frames_cache),
            str(main_fld / f"{model.save_name}_{timestamp()}.mp4"),
        )
    except ValueError:
        pass

    return model.history
