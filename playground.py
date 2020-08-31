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

history = []
for i in range(10):
    u = [100, 101]
    next_x, cost, done, info = env.step(u)
    history.append(next_x)
history = np.vstack(history)

plt.plot(history[:, 0], history[:, 1])


# %%
from sympy import Matrix, diff

xdot_du = Matrix(np.zeros((3, 2)))
for row in range(3):
    for col in range(2):
        xdot_du[row, col] = diff(symbolic.xdot[row], symbolic.tau[col])
        break
    break

# %%
