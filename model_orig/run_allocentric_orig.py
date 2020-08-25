import matplotlib.pyplot as plt
from model_orig import *

config = Config()
env = AlloEnv(config)
model = AlloModel(config)
controller = Controller(config, model)
planner = Planner(config)
runner = Runner(config)

history_x, history_u, history_g = runner.run(env, controller, planner) 

# TODO make this damned thing work
# TODO check the function they had to make sure angles in range
# TODO better plotting


f, ax = plt.subplots(figsize=(10, 10))

ax.plot(history_x[:, 0], history_x[:, 1], lw=1, color='r')

ax.plot(history_g[:, 0], history_g[:, 1], lw=3, color='k', ls='--', alpha=.4, zorder=-1)

plt.show()