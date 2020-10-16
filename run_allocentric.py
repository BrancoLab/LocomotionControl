import pyinspect

pyinspect.install_traceback()


from proj import (
    Model,
    Environment,
    Controller,
    run_experiment,
)


# from proj.plotting.trajectories import plot_trajectory


# model = ModelPolar()
model = Model()
env = Environment(model)
control = Controller(model)


run_experiment(env, control, model, n_secs=1, wrap_up=False)
