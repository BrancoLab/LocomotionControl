from rich import print
from rich.pretty import install
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras import backend as K
from pathlib import Path
from pyinspect.utils import timestamp
import matplotlib.pyplot as plt

from proj.rnn import ControlTask
from proj.paths import rnn_trainig
from proj.utils.progress_bars import train_progress

# TODO make it plot losses
# TODO make it send a slack message

install()

# ---------------------------------- Params ---------------------------------- #
BATCH = 128
T = 2000

EPOCHS = 50
steps_per_epoch = 25

task = ControlTask(dt=5, tau=100, T=T, N_batch=BATCH)
x, y, mask, trial_params = task.get_trial_batch()

STEP = x.shape[1]
N_inputs = x.shape[2]
N_outputs = y.shape[2]

# --------------------------------- Optimizer -------------------------------- #
"""
    Schedule: https://keras.io/api/optimizers/learning_rate_schedules/piecewise_constant_decay/

    Adam https://keras.io/api/optimizers/adam/

    SGD: https://keras.io/api/optimizers/sgd/
    
"""

schedule = PiecewiseConstantDecay([5, 30], [0.01, 0.001, 0.0001])


optimizer = Adam(
    learning_rate=schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)


# ----------------------------------- Model ---------------------------------- #
model = keras.Sequential()
model.add(
    SimpleRNN(
        units=50,
        input_shape=(STEP, N_inputs),
        activation="relu",
        return_sequences=True,
    )
)
model.add(Dense(units=2, activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer=optimizer)

model.summary()


def train_generator():
    while True:
        x, y, mask, trial_params = task.get_trial_batch()
        yield x, y


# ----------------------------------- Train ---------------------------------- #
class CustomCallback(keras.callbacks.Callback):
    def __init__(
        self, epochs, pbar, steps_per_epoch, lr_schedule, *args, **kwargs
    ):
        keras.callbacks.Callback.__init__(self, *args, **kwargs)

        self.loss = 0
        self.step = 0
        self.lr = 0
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pbar = pbar
        self.lr_schedule = lr_schedule

    def on_train_begin(self, logs=None):
        print("[bold magenta]Starting training!")
        self.task_id = self.pbar.add_task(
            "Training",
            start=True,
            total=self.epochs,
            loss=self.loss,
            lr=self.lr,
        )

    def on_train_end(self, logs=None):
        print("[bold green]:tada:  Done!!")

    def on_epoch_begin(self, epoch, logs=None):
        self.lr = K.eval(self.lr_schedule(epoch))
        self.step = 0
        self.pbar.update(
            self.task_id, completed=epoch, loss=self.loss, lr=self.lr,
        )

        self.epoch_task_id = self.pbar.add_task(
            f"Epoch {epoch}",
            start=True,
            total=self.steps_per_epoch,
            loss=self.loss,
            lr=self.lr,
        )

    def on_epoch_end(self, epoch, logs=None):
        self.loss = logs["loss"]
        self.pbar.update(
            self.task_id, completed=epoch, loss=self.loss, lr=self.lr,
        )

        self.pbar.remove_task(self.epoch_task_id)

    def on_train_batch_begin(self, batch, logs=None):
        self.step += 1

    def on_train_batch_end(self, batch, logs=None):
        self.loss = logs["loss"]
        self.pbar.update(
            self.epoch_task_id, completed=batch, loss=self.loss,
        )

    def on_test_begin(self, logs=None):
        # keys = list(logs.keys())
        return

    def on_test_end(self, logs=None):
        # keys = list(logs.keys())
        return

    def on_predict_begin(self, logs=None):
        return

    def on_predict_end(self, logs=None):
        return

    def on_test_batch_begin(self, batch, logs=None):
        # keys = list(logs.keys())
        return

    def on_test_batch_end(self, batch, logs=None):
        # keys = list(logs.keys())
        return

    def on_predict_batch_begin(self, batch, logs=None):
        # keys = list(logs.keys())
        return

    def on_predict_batch_end(self, batch, logs=None):
        # keys = list(logs.keys())
        return


savepath = Path(rnn_trainig).parent / f"keras_model{timestamp()}.h5"
print(f'\n\n[green]Saving model at: "{savepath}"')


with train_progress:

    callback = CustomCallback(
        EPOCHS, train_progress, steps_per_epoch, schedule
    )

    history = model.fit_generator(
        train_generator(),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        verbose=0,
        callbacks=[callback],
    )

model.save(savepath)

# Plot loss
f, ax = plt.subplots(figsize=(12, 8))
ax.plot(history.history["loss"], color="k", lw=2)
ax.set(xlabel="Epoch", ylabel="Loss", title="loss")


# ----------------------------------- Test ----------------------------------- #
# Get an example trial
x, y, mask, trial_params = task.get_trial_batch()

# predict
y_pred = model.predict(x)

f2, axarr = plt.subplots(ncols=4, nrows=4, figsize=(16, 9))

for ax in axarr.flatten():
    x, y, mask, trial_params = task.get_trial_batch()
    ax.plot(y[0, :, :], color="red", label="true", lw=2)
    ax.plot(y_pred[0, :, :], color="blue", label="predicted")

    ax.set(title="Validation", ylabel="Torque", xlabel="Frame")

axarr[0, 0].legend()


plt.show()
