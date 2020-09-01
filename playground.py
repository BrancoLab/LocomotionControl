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

x, y, theta, L, R, m, m_w, d, tau_l, tau_r, v, omega = model.variables.values()



# %%
model = Model()
dt = 0.01
t = np.linspace(0, 20, np.int(20 * (1/dt)))


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
# axarr[0].scatter(model.history['x'], model.history['y'], cmap='bwr', 
#                     c=np.arange(len(model.history['y'])), lw=2)
axarr[0].scatter(model.history['x'][0], model.history['y'][0], marker='D', 
                    lw=1, edgecolors='k', zorder=99)

axarr[0].set(title='xy tracking')

axarr[1].plot(model.history['theta'], lw=2, color='magenta')
axarr[1].set(title='theta')

axarr[2].plot(model.history['v'], lw=2, color='r', label='v')
axarr[2].legend()
axarr[2].set(title='Linear velocity')

axarr[3].plot(model.history['omega'], lw=2, color='b', label='$\omega$')
axarr[3].legend()
axarr[3].set(title='Angular velocity')

_ = axarr[3].set(title='Velocities')
_ = f.tight_layout()


# %%
model.model.jacobian([tau_r, tau_l])
# %%
