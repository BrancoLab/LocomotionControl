# %%
from proj import Model, Environment, Controller, run_experiment, plot_trajectory
import matplotlib.pyplot as plt

# ! TODO  formula I_c for cube!
# TODO once it's mature enough make proper code testing stuff

agent = Model()
env = Environment(agent)
control = Controller(agent)

plot_trajectory(env.reset())
# plt.show()

run_experiment(env, control, agent, n_steps=2000)

