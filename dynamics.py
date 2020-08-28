# %%
"""
    Kinematic and Dynamic model of a 2 wheeled robot with differential drive. 
    Adapted from Dhaouadi et a 2013
"""
# %%
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from collections import namedtuple

# %%
_state = namedtuple('state', 'x, y, theta')
_sides = namedtuple('sides', 'l, r')

_model = namedtuple('model', 'R, L, d, m')

model = _model(
            5,  # wheel rad
            15, # axel half length
            3,  # distance between A and CoM
            2   # mass
        )

# %%
"""
    Kinematics model

    x_dot = Q * [psy_dot_R, psy_dot_L]
"""

def get_dxdt(state, t):
    state = _state(*state)
    r = model.R/2
    a = r * cos(state.theta)
    b = r * sin(state.theta)

    A = np.array(
        [[a, a],
        [b, b],
        [r, -r]
        ]
    )

    return A @ psy_dot

# example of kinematics given fixed psy_dot

psy_dot = _sides(1, 1.1) # ang vel of wheels
state = _state(0, 0, 0) # initial state
t = np.linspace(0, 20)  # simulation time

# integrate and plot
trace = odeint(get_dxdt, state, t)
plt.scatter(trace[:, 0], trace[:, 1], c=trace[:, 2], lw=1, edgecolors='k', cmap='bwr')

# %%

