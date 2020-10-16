import pyinspect as pi

pi.install_traceback()


# import pyinspect
# pyinspect.install_traceback()
import proj
import matplotlib.pyplot as plt
from pathlib import Path

from proj.utils.misc import timestamp
from proj.rnn import ControlTask, RNN


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ---------------------- Set up a basic model ---------------------------
task = ControlTask(dt=10, tau=100, T=2000, N_batch=64)

# get the params passed in and defined in task
network_params = task.get_task_params()

network_params[
    "name"
] = "Control"  # name the model uniquely if running mult models in unison

network_params["N_rec"] = 50  # set the number of recurrent units in the model


model = RNN(network_params)  # instantiate a basic vanilla RNN


# ---------------------- Train a basic model ---------------------------
train_params = {}

# Where to save the model after training.
save_path = (
    Path(proj.paths.rnn_trainig).parent
    / f'training_{network_params["name"]}_{timestamp()}.npz'
)
train_params["save_weights_path"] = save_path

# number of iterations to train for
train_params["training_iters"] = 300000

# Sets learning rate if use
train_params["learning_rate"] = 0.001

# train model to perform pd task
losses, initialTime, trainTime = model.train(task, train_params)

plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Training Iteration")
plt.title("Loss During Training")

plt.show()
# # ---------------------- Test the trained model ---------------------------
# x,target_output,mask, trial_params = pd.get_trial_batch() # get pd task inputs and outputs
# model_output, model_state = model.test(x) # run the model on input x
