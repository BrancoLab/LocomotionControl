import matplotlib.pyplot as plt
from model import *

config = AlloConfig()
env = AlloEnv()
model = AlloModel(config)
controller = Controller(config, model)
planner = Planner(config)
runner = Runner()

history_x, history_u, history_g = runner.run(env, controller, planner) 




f, ax = plt.subplots(figsize=(10, 10))

ax.plot(history_x[:, 0], history_x[:, 1], lw=1, color='r')

ax.plot(history_g[:, 0], history_g[:, 1], lw=3, color='k', ls='--', alpha=.4, zorder=-1)

plt.show()