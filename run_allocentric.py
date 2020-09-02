# %%
from proj import Model, Environment, Controller, run_experiment


# ! TODO PID 
# ! TODO  formula I_c for cube!

# TODO fix failing control goddarnit

# TODO test test test

# TODO once it's mature enough make proper code testing stuff

agent = Model()
env = Environment(agent)
control = Controller(agent)
# %%
run_experiment(env, control, agent)
