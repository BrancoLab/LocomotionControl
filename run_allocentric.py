import matplotlib.pyplot as plt
from model import *
from brainrender.colors import colorMap
import numpy as np

config = Config()
model = AlloModel(config)
env = AlloEnv(config, model)

controller = Controller(config, model)
planner = Planner(config)
runner = Runner(config)

history_x, history_u, history_g, info = runner.run(env, controller, planner) 

# TODO make this damned thing work
# TODO check the function they had to make sure angles in range
# TODO add logger to save results in a better way

# TODO fix this goddamit

f, axarr = plt.subplots(ncols=4, figsize=(12, 10))

# Plot trajectory and goal trajectory

t = history_x.shape[0]
for i in np.arange(t):
    x, y, theta = history_x[i, 0], history_x[i, 1], history_x[i, 2]

    if i % 100 == 0:
        alpha = 1
        axarr[0].plot([x, x + (np.cos(theta)*50)], 
            [y, y + (np.sin(theta)*50)],
            color='k', lw=2)
    else:
        alpha = .4

    color = colorMap(i, name='bwr', vmin=0, vmax=t)
    axarr[0].scatter(x, y, lw=1, color=color, edgecolors='k', alpha=alpha)

# Plot goal trajectory
axarr[0].plot(info['goal_state'][:, 0], info['goal_state'][:, 1], 
                        lw=3, color='k', ls='--', alpha=.4, zorder=-1)

# Ploy initial point and trajectory
axarr[0].scatter(history_x[0, 0], history_x[0, 1], marker='*', s=300, color='m', zorder=100)
axarr[0].plot(history_x[:, 0], history_x[:, 1], lw=1, color='b', zorder=1)

# Plot controls
axarr[1].plot(history_u[:, 0], label='L control', lw=2, color='b')
axarr[1].plot(history_u[:, 1], label='R control', lw=2, color='r')
axarr[1].legend()

# Plot other state vars
axarr[2].plot(np.degrees(history_x[:, 2]), label='Theta', lw=2, color='m')
axarr[2].plot(np.degrees(info['goal_state'][:, 2]), label='Goal Theta', lw=2,
                        color='k', ls='--', alpha=.4, zorder=-1)
axarr[2].legend()

axarr[3].plot(history_x[:, 3], label='Speed', lw=2, color='k')
axarr[3].plot(info['goal_state'][:, 3], label='Goal speed', lw=2,
                        color='k', ls='--', alpha=.4, zorder=-1)
axarr[3].legend()


plt.show()
