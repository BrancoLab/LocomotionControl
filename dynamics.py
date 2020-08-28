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

    A = np.array([
        [a, a],
        [b, b],
        [r, -r]
        ])

    return A @ psy_dot

# example of kinematics given fixed psy_dot

psy_dot = _sides(1, 1.1) # ang vel of wheels
state = _state(0, 0, 0) # initial state
t = np.linspace(0, 20)  # simulation time

# integrate and plot
trace = odeint(get_dxdt, state, t)
plt.scatter(trace[:, 0], trace[:, 1], c=trace[:, 2], lw=1, edgecolors='k', cmap='bwr')

# %%
# TODO implement dynamics 
# TODO bring together
# TODO turn into a complete model for control

# -------------------------------- Kinematics -------------------------------- #

def calc_dxdt(state, nudot):
    state = _state(*state)
    r = model.R/2
    a = r * cos(state.theta)
    b = r * sin(state.theta)

    A = np.array([
        [a, a],
        [b, b],
        [r, -r]
        ])

    return A @ nudot

# --------------------------------- Dynamics --------------------------------- #
def calc_M():
    k = model.R**2 / (4 * model.L**2)
    ml = model.m * model.L**2
    I = 2 * model.m * model.d**2 # ! check if I_c is correct here !!

    M = np.array([
        [k * (ml + I), k * (ml - I)],
        [k * (ml - I), k * (ml + I)],
    ])
    return M

def calc_V(dxdt):
    fact = (model.R ** 2)/2 * model.m * model.d * dxdt.theta
    V = np.array([
        [0, fact],
        [-fact, 0]
    ])
    return V
def calc_B():
    return np.array([
                [1, 0],
                [0, 1]
            ])


def calc_nudot(M, V, B, nu, tau):
    nudot = V @ nu + B @ tau
    return M @ nudot 



# -------------------------------- Simulation -------------------------------- #

def system(state, nu, tau, dxdt):
    # dynamics
    M = calc_M()
    V = calc_V(dxdt)
    B= calc_B()

    nudot = calc_nudot(M, V, B, nu, tau)
    nu = nu + nudot * DT

    # kynematics
    dxdt =  calc_dxdt(state, nudot)
    state = np.array(state) + dxdt * DT

    return _state(*state), _state(*dxdt)



# initialize
DT = .1
duration = 200
t = np.linspace(0, duration, num=int(duration/DT))  # simulation time
state = _state(0, 0, 0) # initialize state
dxdt = _state(0, 0, 0) # initialize dxdt

tau = np.array([np.sin(t), 0*np.cos(t)]) # initialise torques control
nu = np.array([0, 0]) # initialize ang. vels of wheels

# simulate
history = [state]
for n, step in enumerate(t):
    state, dxdt = system(state, nu, tau[:, n], dxdt)
    history.append(state)

x = [h.x for h in history]
y = [h.y for h in history]
theta = [h.theta for h in history]

plt.scatter(x, y, c=theta, cmap='bwr', lw=1, edgecolors='k')


# %%
