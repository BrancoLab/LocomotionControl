# %%
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
from collections import namedtuple
from tqdm import tqdm
from brainrender.colors import colors

from proj.model.model import Model
from proj.utils import wrap_angle

# TODO add moment of inertia of the wheels

# %%
model = Model()

x, y, theta, thetadot, s, L, R, m, d, psi_l, psi_r, psidot_l, \
                    psidot_r, tau_l, tau_r, v, omega, vdot, omegadot = model.variables.values()

model.matrixes['D']

I = 2 * m * d**2

# Define two diff equations
eq1 = Eq(m * vdot - m*d*omega**2, 1/R * (tau_r + tau_l))
eq2 = Eq(I*omegadot + m*d*omega*v, (L/R)*(tau_r - tau_l))

# solve
vdot_sol = solve(eq1, vdot)
omegadot_sol = solve(eq2, omegadot)

# store in matrix
D = Matrix([vdot_sol, omegadot_sol])
D
# %%
model = Model()
dt = 0.01
t = np.linspace(0, 20, np.int(2 * (1/dt)))

# TODO check why cant I set d to zero

u = model._control(1, -1)
history = dict(x=[], y=[], theta=[], s=[])

for it in tqdm(t):
    # if it > 10: 
    #     u = model._control(.11, .1)
    model.step(u)
    # u =  model._control(0, 0)    
# 


f, axarr = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))
axarr = axarr.flatten()

axarr[0].plot(model.history['x'], model.history['y'], color='k', lw=3)
axarr[0].scatter(model.history['x'], model.history['y'], cmap='bwr', 
                    c=np.arange(len(model.history['y'])), lw=2)
axarr[0].scatter(model.history['x'][0], model.history['y'][0], marker='D', 
                    lw=1, edgecolors='k', zorder=99)

axarr[0].set(title='xy tracking')

axarr[1].plot(model.history['theta'], lw=2, color='magenta')
axarr[1].set(title='theta')

axarr[2].plot(model.history['s'], lw=2, color='green')
axarr[2].set(title='speed')

axarr[3].plot(model.history['v'], lw=2, color='r', label='v')
axarr[3].plot(model.history['omega'], lw=2, color='b', label='$\omega$')
axarr[3].legend()

_ = axarr[3].set(title='Velocities')
_ = f.tight_layout()
# %%
model.matrixes['D']
# %%
