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
from models.dynamics.config import Config
from models.dynamics.symbolic import Symbolic

s = Symbolic(Config())
s.xdot
# %%




_state = namedtuple('state', 'x, y, theta')
_sides = namedtuple('sides', 'l, r')

_model = namedtuple('model', 'R, L, d, m')

model = _model(
            s.config.mouse['R'],  # wheel rad
            s.config.mouse['L'], # axel half length
            s.config.mouse['d'],  # distance between A and CoM
            s.config.mouse['m']   # mass
        )

state = _state(0, 0, 0) # initialize state
dxdt = _state(0, 0, 0) # initialize dxdt
tau = [1, 1]
nu = np.array([0, 0]) # initialize ang. vels of wheels

# %%

# -------------------------------- Kinematics -------------------------------- #

def calc_dxdt(state, nudot):
    Q = calc_Q(state)
    return Q @ nudot

def calc_Q(state):
    state = _state(*state)
    r = model.R/2
    a = r * cos(state.theta)
    b = r * sin(state.theta)

    Q = np.array([
        [a, a],
        [b, b],
        [r, -r]
        ]).astype(np.float64)
    return Q

# --------------------------------- Dynamics --------------------------------- #

def calc_M_inv():
    k = model.R**2 / (4 * model.L**2)
    ml = model.m * model.L**2
    I = 2 * model.m * model.d**2 # ! check if I_c is correct here !!

    M = np.array([
        [k * (ml + I), k * (ml - I)],
        [k * (ml - I), k * (ml + I)],
    ]).astype(np.float64)
    return np.linalg.inv(M)

def calc_V(dxdt):
    fact = (model.R ** 2)/2 * model.m * model.d * dxdt.theta
    V = np.array([
        [0, fact],
        [-fact, 0]
    ]).astype(np.float64)
    return V

def calc_B():
    return np.array([
                [1, 0],
                [0, 1]
            ]).astype(np.float64)


def calc_nudot(M_inv, V, B, nu, tau):
    nudot = M_inv  @  (B @ tau - V @ nu) 
    return nudot 



# -------------------------------- Simulation -------------------------------- #
values = dict(
    R = s.config.mouse['R'],
    L = s.config.mouse['L'],
    m = s.config.mouse['m'],
    d = s.config.mouse['d'],
    theta = state.theta, 
    thetadot = dxdt.theta,
    tau_r = tau[0],
    tau_l = tau[1],
    eta_r = nu[0],
    eta_l = nu[1],
)

def get_dxdt_fancy():
    return  s.eval(s.xdot, values)



def get_dxdt_trad(state, nu, tau, dxdt):
    Q = calc_Q(state)

    # dynamics
    M_inv = calc_M_inv()
    V = calc_V(dxdt)
    B = calc_B()

    # kynematics
    return  Q @ M_inv @ (B @ tau - V @ nu)

    

trad = get_dxdt_trad(state, nu, tau, dxdt)
fancy = get_dxdt_fancy()

print('Traditional: \n', trad)
fancy

