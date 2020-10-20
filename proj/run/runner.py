import numpy as np
from rich import print

from loguru import logger

from pyinspect.utils import timestamp

from proj.utils.progress_bars import progress
from pyinspect._colors import mocassin, salmon, green, lilla


def compare_controllers(curr_x, g_xs, main_controller_u, *controllers):

    print(
        f"[{mocassin}]Main controllers solution: [bold {green}]{[int(round(x)) for x in main_controller_u]}[/bold {green}]"
        + f"   sum: {np.sum(main_controller_u):.0f}  diff: {np.diff(main_controller_u)[0]:.0f}"
    )

    for con in controllers:
        sol = con.obtain_sol(curr_x, g_xs)
        print(
            f"[{mocassin}]   alternative controller: [{salmon}]{[int(round(x)) for x in sol]}"
            + f"   sum: {np.sum(sol):.0f}  diff: {np.diff(sol)[0]:.0f}"
        )
    print(
        f"[{mocassin}]               difference: [{lilla}]{[int(round(x-y)) for x,y in zip(sol, main_controller_u)]}"
    )
    print("\n\n")


# run
def run_experiment(
    environment,
    controller,
    model,
    n_secs=8,
    frames_folder=None,
    wrap_up=True,
    extra_controllers=None,
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

    # reset things
    trajectory = environment.reset()
    if trajectory is None:
        logger.info("Failed to get a valid trajectory")
        environment.failed()
        return
    model.reset()

    # Get number of steps
    n_steps = int(n_secs / model.dt)
    print(
        f"\n\n[bold  green]Starting simulation with {n_steps} steps [{n_secs}s at {model.dt} s/step][/bold  green]"
    )

    # RUN
    start = timestamp(just_time=True)
    with progress:
        task_id = progress.add_task("running", start=True, total=n_steps)

        for itern in range(n_steps):
            try:
                progress.advance(task_id, 1)

                curr_x = np.array(model.curr_x)

                # plan
                g_xs = environment.plan(curr_x, trajectory, itern)
                if g_xs is None:
                    break  # we're done here

                # obtain sol
                u = controller.obtain_sol(curr_x, g_xs)

                # get current cost
                try:
                    environment.curr_cost = controller.calc_step_cost(
                        np.array(model.curr_x), u, g_xs[0, :]
                    )
                except AttributeError:
                    environment.curr_cost = dict(control=0, state=0, total=0)

                if extra_controllers is not None:
                    compare_controllers(curr_x, g_xs, u, *extra_controllers)

                # step
                model.step(
                    u, g_xs[1, :]
                )  # g_xs[1, :] is used to keep a history of state delta

                # update world
                environment.itern = itern
                environment.update_world(g_xs, elapsed=itern * model.dt)

                # log status once a (simulation) second
                if itern % int(1 / model.dt) == 0:
                    logger.info(
                        f"Iteration {itern}/{n_steps}. Current cost: {environment.curr_cost}."
                    )

                # Check if we're done
                if environment.isdone(model.curr_x, trajectory):
                    logger.info("environment says we're DONE")
                    break
                if environment.stop:
                    logger.info("environment says STOP")
                    break

            except Exception as e:
                logger.exception(
                    f"Failed to take next step in simulation.\nError: {e}\n\n"
                )
                break

    logger.info(f"Started at {start}, finished at {timestamp(just_time=True)}")

    if wrap_up:
        try:
            environment.conclude()
        except Exception as e:
            logger.info(f"Failed to run environment.conclude(): {e}")
            environment.failed()
            return
