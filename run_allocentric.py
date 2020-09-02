# %%
from proj import Model, Environment, Controller, run_experiment


# ! TODO  formula I_c for cube!

# TODO test test test

# TODO once it's mature enough make proper code testing stuff

agent = Model()
env = Environment(agent)
control = Controller(agent)
# %%
run_experiment(env, control, agent, n_steps=2000)
