# %%
import matplotlib.pyplot as plt
from brainrender.colors import colorMap
import numpy as np

from models.dynamics import *

config = Config()
symbolic = Symbolic(config)

# %%
x, y, R, theta, thetadot, L, m, d, taur, taul, etar, etal = symbolic.symbols.values()
vals  = symbolic.symbols.values()
# %%
from sympy import lambdify

fun = lambdify(vals, symbolic.xdot_dx)


# %%
