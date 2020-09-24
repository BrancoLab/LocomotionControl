from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
)
import numpy as np
from rich import print
from rich.text import Text
import logging


class SpeedColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        if task.speed is None:
            return Text("no speed")
        else:
            return Text(f"{task.speed:.3f} steps/s")


progress = Progress(
    TextColumn("[bold magenta]Step {task.completed}/{task.total}"),
    SpeedColumn(),
    # "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    "•",
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeRemainingColumn(),
)


# run
def run_experiment(
    environment, controller, model, n_secs=30, frames_folder=None,
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
    log = logging.getLogger("rich")

    # reset things
    trajectory = environment.reset()
    if trajectory is None:
        log.info("Failed to get a valid trajectory")
        environment.failed()
        return

    model.reset()

    # Get number of steps
    n_steps = int(n_secs / model.dt)
    print(
        f"\n\n[bold  green]Starting simulation with {n_steps} steps [{n_secs}s at {model.dt} s/step][/bold  green]"
    )

    # RUN
    with progress:
        task_id = progress.add_task("running", start=True, total=n_steps)

        for itern in range(n_steps):
            try:
                progress.advance(task_id, 1)

                curr_x = np.array(model.curr_x)

                # plan
                g_xs = environment.plan(curr_x, trajectory, itern)

                # obtain sol
                u = controller.obtain_sol(curr_x, g_xs)

                # step
                model.step(u)

                # get current cost
                environment.curr_cost = controller.calc_step_cost(
                    np.array(model.curr_x), u, g_xs[0, :]
                )

                # update world
                environment.itern = itern
                environment.update_world(g_xs, elapsed=itern * model.dt)

                # log status once a second
                if itern % int(1 / model.dt) == 0:
                    log.info(
                        f"Iteration {itern}/{n_steps}. Current cost: {environment.curr_cost}."
                    )

                # Check if we're done
                if environment.isdone(model.curr_x, trajectory):
                    log.info("environment says we're DONE")
                    break
                if environment.stop:
                    log.info("environment says STOP")
                    break
            except Exception as e:
                logging.error(
                    f"Failed to take next step in simulation.\n error {e}"
                )
                break

    log.info(f"Terminated after {itern} iterations.")

    # save data and close stuff
    environment.conclude()
