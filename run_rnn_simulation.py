import pyinspect

pyinspect.install_traceback()

from proj import (
    Model,
    Environment,
    RNNController,
    run_experiment,
    Controller,
)
from proj.rnn import ControlTask

# ? setup other stuff
model = Model()
model.LIVE_PLOT = True
env = Environment(model)


# ? Setup RNN controller
# fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\control\\RNN\\working_model"
fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/RNN/working_model"
task = ControlTask(dt=10, tau=100, T=2000, N_batch=1)

# get the params passed in and defined in task
network_params = task.get_task_params()

network_params[
    "name"
] = "Control"  # name the model uniquely if running mult models in unison

network_params["N_rec"] = 50  # set the number of recurrent units in the model

control = RNNController(model, fld, network_params)
alt_control = Controller(model)


# ? RUN
run_experiment(
    env,
    alt_control,
    model,
    n_secs=3,
    wrap_up=False,
    extra_controllers=[control],
)
