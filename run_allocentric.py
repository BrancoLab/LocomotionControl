# %%
from proj import (
    Model,
    Environment,
    Controller,
    run_experiment,
    # ModelPolar,
)

# from proj.plotting.trajectories import plot_trajectory

# TODO get trials from M4 catwalk and M6 catwalk only.
# TODO trajectory: get moved awaay from start location as start frame
# TODO find metrics to say if it was good/bad?

# model = ModelPolar()
model = Model()
env = Environment(model)
control = Controller(model)

# plot_trajectory(env.reset())

run_experiment(env, control, model)
