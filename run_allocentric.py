# %%
from proj import (
    Model,
    Environment,
    Controller,
    run_experiment,
    # ModelPolar,
)


# ! TODO  formula I_c for cube!
# TODO energy considerations, you should need more force to accelerate when going faster?

# model = ModelPolar()
model = Model()
env = Environment(model)
control = Controller(model)


run_experiment(env, control, model, plot=False)
