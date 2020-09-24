# Set up logging
import logging

for module in ["matplotlib", "pandas", "numpy"]:
    requests_logger = logging.getLogger("matplotlib")
    requests_logger.setLevel(logging.ERROR)

# import stuff
from proj.model.config import Config
from proj.model.model import Model
from proj.model.model_polar import ModelPolar

from proj.run.runner import run_experiment
from proj.run.runner_manual import run_manual

from proj.environment.environment import Environment

from proj.control.control import Controller

from proj.plotting.trajectories import plot_trajectory, plot_trajectory_polar
from proj.plotting.results import plot_results
