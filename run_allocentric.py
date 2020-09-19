# %%
from proj import (
    Model,
    Environment,
    Controller,
    run_experiment,
    # ModelPolar,
)

# TODO get trials from M4 catwalk and M6 catwalk only.
# TODO save cost history and add to summary plot

# model = ModelPolar()
model = Model()
env = Environment(model)
control = Controller(model)

# plot_trajectory(env.reset())

run_experiment(env, control, model)
#
