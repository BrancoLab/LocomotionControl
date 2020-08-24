# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.integrate import odeint
from brainrender.colors import colorMap
from math import cos, sin

# %%

# ------------------------------ Utils and vars ------------------------------ #

state = namedtuple('state', 'x, y, theta, v')
control = namedtuple('control', 'L, R')

def plot_state(st, ax, i, color):
    if i % 10 == 0:
        alpha = 1
        ax.plot([st.x, st.x + (cos(st.theta)*50)], 
            [st.y, st.y + (sin(st.theta)*50)],
            color='k', lw=2)

    else:
        alpha = .2

    ax.scatter(st.x, st.y, s=50, color=color, alpha=alpha, lw=2, edgecolors='k')
    

# %%
# -------------------------------- Parameters -------------------------------- #
m = 10 # mass
u = control(4.8, 5.2)
x = state(0, 0, 0, 0)


# ----------------------------------- model ---------------------------------- #
def model(x, t):
    # Define arbitrary control
    if t < 20:
        u = control(4, 4.1)
    else:
        u = control(4.1, 4)

    # Compute x_dot
    x = state(*x)
    dxdt = [
        x.v * cos(x.theta),
        x.v * sin(x.theta),
        (u.R - u.L) / m,
        (u.R + u.L)/m + (1 - (u.R - u.L)/(u.R + u.L))
    ]
    return dxdt


# -------------------------------- SIMULATION -------------------------------- #
t = np.linspace(0, 50, 51)
x = odeint(model, x, t)

f, ax = plt.subplots(figsize=(8, 8))

# Plot results
for i in t:
    color = colorMap(i, name='bwr', vmin=0, vmax=np.max(t))
    plot_state(state(*x[int(i), :]), ax, i, color)


# %%


# %%
