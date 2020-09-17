from proj import (
    Model,
    Environment,
    Controller,
    run_experiment,
)


model = Model()
env = Environment(model, winstor=True)
control = Controller(model)


run_experiment(env, control, model, plot=False, n_secs=0.2)
