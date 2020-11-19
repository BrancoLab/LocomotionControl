import pyinspect
from rich.pretty import install

install()
pyinspect.install_traceback(hide_locals=True)

# import stuff
from proj.model.config import Config
from proj.model.model import Model
from proj.run.runner import run_experiment
from proj.environment.environment import Environment
from proj.control.control import Controller
from proj.plotting.results import plot_results
