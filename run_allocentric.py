# %%

from proj import (
    # Model,
    Environment,
    Controller,
    run_experiment,
    ModelPolar,
    # plot_trajectory,
    # run_manual,
)

# ! TODO  formula I_c for cube!
# ! energy considerations, you should need more force to accelerate when going faster?

# TODO fix orientation at start of trajectory

model = ModelPolar()
env = Environment(model)
control = Controller(model)

run_experiment(env, control, model)
