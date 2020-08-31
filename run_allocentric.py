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

# TODO xdot_du doesn't depend have tau or nu terms?

# TODO figure out why model sucks
# TODO something about how it computes future steps fucks up cost calculation


# TODO make speed back into state

# %%
