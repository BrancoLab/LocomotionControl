# %%
import matplotlib.pyplot as plt
from brainrender.colors import colorMap
import numpy as np

from models.dynamics import *

config = Config()
symbolic = Symbolic(config)
model = Model(config, symbolic)
env = Environment(config, model)

controller = Controller(config, model)
planner = Planner(config)
runner = Runner(config, interactive_plot)

# %%
history_x, history_u, history_g, info = runner.run(env, controller, planner) 

# TODO separate the kinematic and dynamic models and apply control to each individually.

# TODO why is it going the wrong way stupid

# TODO make it faster (e.g. cupy?)
# TODO add friction?

# TODO add code to cache results + make plots/gifs out of it

# %%
