# %%

import pyinspect

pyinspect.install_traceback()


from proj import (
    Model,
    Environment,
    Controller,
    run_experiment,
    # ModelPolar,
)


# from proj.plotting.trajectories import plot_trajectory


# model = ModelPolar()
model = Model()
env = Environment(model)
control = Controller(model)

# plot_trajectory(env.reset())

# %%
run_experiment(env, control, model, n_secs=0.01)


# %%
