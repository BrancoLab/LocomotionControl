import pyinspect

pyinspect.install_traceback()

from pathlib import Path
from proj import (
    Model,
    Environment,
    RNNController,
    run_experiment,
    Controller,
)
from proj import paths

# ? setup other stuff
model = Model()
model.LIVE_PLOT = True
env = Environment(model)


# ? Setup RNN controller
fld = Path(paths.rnn) / "RNN_100units_good"
control = RNNController(fld)
alt_control = Controller(model)

# ? RUN
run_experiment(
    env,
    control,
    model,
    n_secs=3,
    wrap_up=False,
    # extra_controllers=[alt_control],
)
