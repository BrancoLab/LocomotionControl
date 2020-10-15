import pyinspect as pi

pi.install_traceback()


# import pyinspect
# pyinspect.install_traceback()
import proj
import matplotlib.pyplot as plt
from pathlib import Path

from proj.utils.misc import timestamp
from proj.rnn.task import ControlTask

from psychrnn.backend.models.basic import Basic


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ---------------------- Set up a basic model ---------------------------
task = ControlTask(dt=10, tau=100, T=3000, N_batch=128)
network_params = (
    task.get_task_params()
)  # get the params passed in and defined in task
network_params[
    "name"
] = "Control"  # name the model uniquely if running mult models in unison
network_params["N_rec"] = 100  # set the number of recurrent units in the model

save_path = (
    Path(proj.paths.rnn_trainig).parent
    / f'training_{network_params["name"]}_{timestamp()}.npz'
)


model = Basic(network_params)  # instantiate a basic vanilla RNN
# ---------------------- Train a basic model ---------------------------
train_params = {}
train_params[
    "save_weights_path"
] = save_path  # Where to save the model after training. Default: None
train_params[
    "training_iters"
] = 300000  # number of iterations to train for Default: 50000
train_params[
    "learning_rate"
] = 0.005  # Sets learning rate if use default optimizer Default: .001


losses, initialTime, trainTime = model.train(
    task, train_params
)  # train model to perform pd task

plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Training Iteration")
plt.title("Loss During Training")

plt.show()
# # ---------------------- Test the trained model ---------------------------
# x,target_output,mask, trial_params = pd.get_trial_batch() # get pd task inputs and outputs
# model_output, model_state = model.test(x) # run the model on input x
