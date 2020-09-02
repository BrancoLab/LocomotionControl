# %%
from proj import Model, Environment, Controller, run_experiment

# TODO check angles in trajectory
# TODO check why speed won't be accurate -> | IT HAS TO DO WITH DT, larger DT makes it more precise

# ! TODO  formula I_c for cube!

# TODO test test test

# TODO once it's mature enough make proper code testing stuff

agent = Model()
env = Environment(agent)
control = Controller(agent)
# %%
run_experiment(env, control, agent, n_steps=2000)

# TODO make it run on HPC
# TODO run on a number of different trajectories
