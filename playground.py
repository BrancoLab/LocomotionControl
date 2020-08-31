# %%
import matplotlib.pyplot as plt
from brainrender.colors import colorMap
import numpy as np

from models.dynamics import *

config = Config()
symbolic = Symbolic(config)
model = Model(config, symbolic)
env = Environment(config, model)



# %%
env.reset()
env.curr_x = np.zeros(4)

history = []
for i in range(1000):
    u = [2, -2]
    next_x, cost, done, info = env.step(u)
    history.append(next_x)
history = np.vstack(history)

f, axarr = plt.subplots(ncols=2)
axarr[0].plot(history[:, 0], history[:, 1])
axarr[1].plot(history[:, 2])
axarr[1].plot(history[:, 3])


# %%
from sympy import Matrix, diff

xdot_du = Matrix(np.zeros((3, 2)))
for row in range(3):
    for col in range(2):
        xdot_du[row, col] = diff(symbolic.xdot[row], symbolic.tau[col])
        break
    break

# %%
