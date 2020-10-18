from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


import numpy as np
from pyinspect.utils import timestamp
from pyinspect._colors import orange, lightorange
import matplotlib.pyplot as plt
from rich.progress import track

from fcutils.plotting.utils import clean_axes, save_figure

from proj.rnn._utils import RNNLog
from proj.rnn.task import ControlTask
from proj.utils.progress_bars import train_progress, CustomCallback
from proj.utils.slack import send_slack_message


_layers = dict(simpleRNN=SimpleRNN, dense=Dense,)


class RNNTrainer(RNNLog):
    def __init__(self,):
        super(RNNTrainer, self).__init__()

        # Get a few useful variabels
        self.task = ControlTask(
            dt=self.config["dt"],
            tau=self.config["tau"],
            T=self.config["T"],
            N_batch=self.config["BATCH"],
        )

        x, y, mask, trial_params = self.task.get_trial_batch()

        self.STEP = x.shape[1]
        self.N_inputs = x.shape[2]
        self.N_outputs = y.shape[2]
        self.batch_input_shape = x.shape

    def _make_dense_layer(self, layer_params):
        return _layers[layer_params["name"]](
            units=layer_params["units"],
            activation=layer_params["activation"],
            name="Dense",
        )

    def _make_rnn_layer(self, layer_params):
        return _layers[layer_params["name"]](
            units=layer_params["units"],
            activation=layer_params["activation"],
            input_shape=(self.STEP, self.N_inputs),
            batch_input_shape=self.batch_input_shape,
            return_sequences=True,
            name="Recurrent",
        )

    def make_model(self):
        self.log.spacer(2)
        self.log.add(f"[b {orange}]Creating model")

        # --------------------------------- scheduler -------------------------------- #

        schedule = PiecewiseConstantDecay(
            self.config["lr_schedule"]["boundaries"],
            self.config["lr_schedule"]["values"],
        )
        self.log.add(
            f'Learning rate scheduler: {self.config["lr_schedule"]["name"]}\n bounds: {self.config["lr_schedule"]["boundaries"]} - vals: {self.config["lr_schedule"]["values"]} '
        )

        # --------------------------------- optimzer --------------------------------- #

        if self.config["optimizer"] != "Adam":
            raise NotImplementedError(
                f"Needs to be setup to work with optimizer {self.config['optimizer']}"
            )
        optimizer = Adam(
            learning_rate=schedule,
            name="Adam",
            clipvalue=self.config["clipvalue"],
            amsgrad=self.config["amsgrad"],
        )

        self.log.add(
            f'Optimizer: {self.config["optimizer"]} with clipvalue: {self.config["clipvalue"]}'
        )
        self.log.add(f'Loss function: {self.config["loss"]}')

        # ----------------------------------- model ---------------------------------- #

        self.log.spacer(2)
        self.log.add(f"[{orange}]Layers")
        model = keras.Sequential()
        for n, layer in enumerate(self.config["layers"]):
            if layer["name"] == "dense":
                l = self._make_dense_layer(layer)
            else:
                l = self._make_rnn_layer(layer)

            model.add(l)
            self.log.add(
                f'[green]Layer {n}[/green]  --  [b {lightorange}]{layer["name"]}[/b {lightorange}] - [blue]{layer["units"]}[/blue] units - [green]{layer["activation"]}[/green] activation - stateful {layer["stateful"]}'
            )

        # ---------------------------------- compile --------------------------------- #

        model.compile(loss=self.config["loss"], optimizer=optimizer)

        self.callback = CustomCallback(
            self.config["EPOCHS"],
            train_progress,
            self.config["steps_per_epoch"],
            schedule,
            self.log,
        )

        return model

    def make_data(self):
        print("[bold magenta]Creating training data...")

        X, Y = [], []
        if self.config["EPOCHS"] < 101:
            for i in track(
                range(self.config["EPOCHS"]), description="Preparing data..."
            ):
                x, y, mask, trial_params = self.task.get_trial_batch()
                X.append(x)
                Y.append(y)

            return np.concatenate(X), np.concatenate(Y)

        # If it's a lot of epochs it's faster to just get the whole thing at once
        self.task = ControlTask(
            dt=self.config["dt"],
            tau=self.config["tau"],
            T=self.config["T"],
            N_batch=2000,
        )
        x, y, mask, trial_params = self.task.get_trial_batch()
        return x, y

    def train(self):
        self.save_config()

        model = self.make_model()
        x, y = self.make_data()

        start = timestamp(just_time=True)
        self.log.spacer(2)
        self.log.add(f"[b {orange}]Starting training at: {start}")

        with train_progress:
            history = model.fit(
                x,
                y,
                steps_per_epoch=self.config["steps_per_epoch"],
                epochs=self.config["EPOCHS"],
                verbose=0,
                callbacks=[self.callback],
                use_multiprocessing=True,
                workers=8,
                # validation_split=0.2,
                sample_weight=np.array(self.config["sample_weight"])
                if self.config["sample_weight"] is not None
                else None,
            )

        send_slack_message(
            f"""
                            \n
                            Completed RNN training
                            Start time: {start}
                            End time: {timestamp(just_time=True)}
                            Final loss: {history.history['loss'][-1]:.6f}
                        """
        )
        self.log.add(
            f"[dim {orange}]Training finished: {timestamp(just_time=True)}"
        )

        self.log.spacer(2)
        self.log.add(f"Saving model at: {self.rnn_weights_save_path}")
        model.save(self.rnn_weights_save_path)

        self.log.print()
        self.save_log(self.log)

        self.save_config()

        self.plot_weights(model)
        self.plot_training_history(history)
        self.plot_training_evaluation(model)
        plt.show()

    def plot_weights(self, model):

        f, axarr = plt.subplots(
            ncols=3, nrows=len(model.layers), figsize=(16, 9)
        )

        for ln, layer in enumerate(model.layers):
            weights = layer.get_weights()

            for ax, w, ttl in zip(
                axarr[ln, :], weights, ["in", "recurrent", "bias"]
            ):
                if len(w.shape) == 1:
                    w = w.reshape((-1, 1))
                ax.imshow(w, cmap="bwr")
                ax.set(
                    title=f"{layer.name}  -- "
                    + ttl
                    + f" - vmin:{w.min():.2f} - vmax:{w.max():.2f}"
                )
                ax.axis("equal")
                ax.axis("off")

        clean_axes(f)
        save_figure(f, self.folder / "weights", verbose=False)

    def plot_training_history(self, history):
        f, ax = plt.subplots(figsize=(16, 9))

        ax.plot(history.history["loss"], color="k", label="training loss")

        try:
            ax.plot(
                history.history["val_loss"],
                color="red",
                label="validation loss",
            )
        except KeyError:
            pass

        ax.legend()
        ax.set(xlabel="# epoch", ylabel="loss")

        clean_axes(f)
        save_figure(f, self.folder / "loss", verbose=False)

    def plot_training_evaluation(self, model):
        # Get an example trial
        x, y, mask, trial_params = self.task.get_trial_batch()

        # predict
        y_pred = model.predict(x)

        f2, axarr = plt.subplots(
            ncols=4, nrows=4, figsize=(16, 9), sharex=True, sharey=True
        )

        for n, ax in enumerate(axarr.flatten()):
            ax.plot(y[n, :, 0], ls="--", lw=2, color="red", label="true R")
            ax.plot(y[n, :, 1], ls="--", lw=2, color="magenta", label="true L")

            ax.plot(y_pred[n, :, 0], color="blue", label="predicted R")
            ax.plot(y_pred[n, :, 1], color="green", label="predicted L")

        ax.set(ylabel="Torque", xlabel="Frame")  # , ylim=[0.2, 0.8])
        axarr[0, 0].legend()

        clean_axes(f2)
        save_figure(f2, self.folder / "validation", verbose=False)
