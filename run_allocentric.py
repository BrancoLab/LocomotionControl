import matplotlib.pyplot as plt
from brainrender.colors import colorMap
import numpy as np

from models import allo_two_wheels as _model

config = _model.Config()
model = _model.Model(config)
env = _model.Environment(config, model)

controller = _model.Controller(config, model)
planner = _model.Planner(config)
runner = _model.Runner(config, _model.utils.interactive_plot)

history_x, history_u, history_g, info = runner.run(env, controller, planner) 

# TODO check if L/R turns are inverted
# TODO make it work with realistic mass goddarmit
# TODO add things like inertia, drag and stuff... laws of motion baby

# TODO look into way to enfore smooth controls | might come from better dynamics