from tqdm import tqdm


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

    # RUN
    for itern in tqdm(range(n_steps)):
        # plan
        g_xs = environment.plan(model.curr_x, trajectory)

        # obtain sol
        u = controller.obtain_sol(model.curr_x, g_xs)

        # step
        next_x, cost, done, info = model.step(u)

    return model.history