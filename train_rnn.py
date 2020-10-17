import pyinspect as pi

pi.install_traceback()


from pyinspect.utils import timestamp
import proj
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

from proj.rnn import ControlTask, RNN
from proj.utils.slack import send_slack_message


# ---------------------- Set up a basic model ---------------------------
task = ControlTask(dt=5, tau=100, T=2000, N_batch=128)

# get the params passed in and defined in task
network_params = task.get_task_params()

network_params[
    "name"
] = "Control"  # name the model uniquely if running mult models in unison

network_params["N_rec"] = 50  # set the number of recurrent units in the model
network_params["transfer_function"] = tf.nn.sigmoid

model = RNN(network_params)  # instantiate a basic vanilla RNN


# ---------------------- Train a basic model ---------------------------
train_params = {}

# Where to save the model after training.
fname = f'training_{network_params["name"]}_{timestamp()}.npz'
save_path = Path(proj.paths.rnn_trainig).parent / fname
train_params["save_weights_path"] = save_path

# number of iterations to train for
train_params["training_iters"] = 50000

# Sets learning rate if use
train_params["learning_rate"] = 0.001

# train model to perform pd task
start = timestamp(just_time=True)
losses, initialTime, trainTime = model.train(task, train_params)

send_slack_message(
    f"""
                    \n
                    Completed RNN training
                    Start time: {start}
                    End time: {timestamp(just_time=True)}
                    Final loss: {losses[-1]:.3e}
                """
)

plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Training Iteration")
plt.title("Loss During Training")

plt.show()
# # ---------------------- Test the trained model ---------------------------
# x,target_output,mask, trial_params = pd.get_trial_batch() # get pd task inputs and outputs
# model_output, model_state = model.test(x) # run the model on input x
