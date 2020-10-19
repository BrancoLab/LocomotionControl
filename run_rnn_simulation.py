import pyinspect

pyinspect.install_traceback()

from proj import (
    Model,
    Environment,
    RNNController,
    run_experiment,
    Controller,
)

# ? setup other stuff
model = Model()
model.LIVE_PLOT = True
env = Environment(model)


# ? Setup RNN controller
# fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\control\\RNN\\RNN_100units_good"
fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/control/RNN/RNN_100units_good"

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
