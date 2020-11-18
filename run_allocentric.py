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
model = Model(trial_n=1)
env = Environment(model)
control = Controller(model)


run_experiment(env, control, model, n_secs=2, wrap_up=False)
