import pyinspect
from rich.pretty import install

install()
pyinspect.install_traceback()

# Set up logging
# from proj.utils.logging import log

# import stuff
from proj.model.config import Config
from proj.model.model import Model
from proj.model.model_polar import ModelPolar
from proj.run.runner import run_experiment
from proj.environment.environment import Environment
from proj.control.control import Controller, RNNController
from proj.plotting.results import plot_results
