# %%
from proj import (
    Model,
    Environment,
    Controller,
    run_experiment,
    # ModelPolar,
)

# from proj.plotting.trajectories import plot_trajectory

# ! TODO  formula I_c for cube!
# TODO energy considerations, you should need more force to accelerate when going faster?

# model = ModelPolar()
model = Model()
env = Environment(model)
control = Controller(model)

# plot_trajectory(env.reset())

run_experiment(env, control, model)
#
