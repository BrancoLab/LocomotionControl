from rich import print
from rich.pretty import install
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from pathlib import Path
from pyinspect.utils import timestamp
import matplotlib.pyplot as plt


from fcutils.plotting.utils import clean_axes, save_figure

from proj.rnn import ControlTask
from proj.paths import rnn_trainig
from proj.utils.progress_bars import train_progress, CustomCallback
from proj.utils.slack import send_slack_message


install()

# ---------------------------------- Params ---------------------------------- #
BATCH = 64
T = 2000

EPOCHS = 60
steps_per_epoch = 50

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

schedule = PiecewiseConstantDecay([20, 40], [0.001, 0.0001, 0.00001])


optimizer = Adam(learning_rate=schedule, name="Adam", clipvalue=0.5)


# ----------------------------------- Model ---------------------------------- #
model = keras.Sequential()
model.add(
    SimpleRNN(
        units=250,
        input_shape=(STEP, N_inputs),
        batch_input_shape=x.shape,
        activation="relu",
        return_sequences=True,
        stateful=True,
    )
)
model.add(Dense(units=2, activation="relu"))
model.compile(loss="mean_squared_error", optimizer=optimizer)

model.summary()


def train_generator():
    while True:
        x, y, mask, trial_params = task.get_trial_batch()
        yield x, y


# ----------------------------------- Train ---------------------------------- #
savepath = Path(rnn_trainig).parent / f"keras_model{timestamp()}.h5"
print(f'\n\n[green]Saving model at: "{savepath}"')


callback = CustomCallback(EPOCHS, train_progress, steps_per_epoch, schedule)
start = timestamp(just_time=True)


with train_progress:
    history = model.fit_generator(
        train_generator(),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        verbose=0,
        callbacks=[callback],
    )

model.save(savepath)

send_slack_message(
    f"""
                    \n
                    Completed RNN training
                    Start time: {start}
                    End time: {timestamp(just_time=True)}
                    Final loss: {history.history['loss'][-1]:.3e}
                """
)

# Plot loss
f, ax = plt.subplots(figsize=(12, 8))
ax.plot(history.history["loss"], color="k", lw=2)
ax.set(xlabel="Epoch", ylabel="Loss", title="loss")


# ----------------------------------- Test ----------------------------------- #
# Get an example trial
x, y, mask, trial_params = task.get_trial_batch()

# predict
y_pred = model.predict(x)

f2, axarr = plt.subplots(
    ncols=4, nrows=4, figsize=(16, 9), sharex=True, sharey=True
)

for n, ax in enumerate(axarr.flatten()):
    ax.plot(y[n, :, :], color="red", label="true", lw=2)
    ax.plot(y_pred[n, :, :], color="blue", label="predicted")

    ax.set(ylabel="Torque", xlabel="Frame", ylim=[0.2, 0.8])

axarr[0, 0].legend()

clean_axes(f)
clean_axes(f2)
save_figure(f, Path(rnn_trainig).parent / "loss")
save_figure(f2, Path(rnn_trainig).parent / "validation")
plt.show()
