# %%
from proj import Model, Environment, Controller, run_experiment
# %%

# TODO env.plan make it more accurate, make sure it uses only X,Y for selection

agent = Model()
env = Environment(agent)
control = Controller(agent)
# %%
run_experiment(env, control, agent)
