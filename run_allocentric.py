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

history_x, history_u, history_g, info = runner.run(env, controller, planner) 

# ! TODO check why I'm getting nan in f_x and f_u
# ! TODO it's not giving errors but no control
# TODO remove warnings