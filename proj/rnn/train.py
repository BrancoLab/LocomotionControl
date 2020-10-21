import numpy as np
from pyinspect.utils import timestamp
from pyinspect._colors import orange
import matplotlib.pyplot as plt
from rich.progress import track
from rich import print

from fcutils.plotting.utils import clean_axes, save_figure

from proj.rnn.rnn import ControlRNN
from proj.utils.progress_bars import train_progress
from proj.utils.slack import send_slack_message
from proj.rnn import ControlTask


class RNNTrainer(ControlRNN):
    """
        Main class taking care of training RNNs
        to fit the predictions of the iLQR algorithm.
    """

    def __init__(self, *args, wrap_up=True, **kwargs):
        ControlRNN.__init__(self, *args, **kwargs)
        self._wrap_up = wrap_up

        self.eval_task = ControlTask(
            dt=self.config["dt"],
            tau=self.config["tau"],
            T=self.config["T"],
            N_batch=self.config["BATCH"],
            test_data=True,
            **kwargs,
        )

    def make_data(self):
        print(
            f"[bold magenta]Creating training data... [dim]from: {self.dataset_test_path.parent.name}"
        )

        X, Y = [], []
        # if self.config["EPOCHS"] < 101:
        for i in track(
            range(self.config["EPOCHS"]), description="Preparing data..."
        ):
            x, y, mask, trial_params = self.task.get_trial_batch()
            X.append(x)
            Y.append(y)

        return np.concatenate(X), np.concatenate(Y), mask

        # If it's a lot of epochs it's faster to just get the whole thing at once
        # self.task.N_batch = self.task._n_trials
        # x, y, mask, trial_params = self.task.get_trial_batch()
        # return x, y, mask

    def train(self):
        self.save_config()

        model = self.make_model()
        x, y, mask = self.make_data()

        start = timestamp(just_time=True)
        self.log.spacer(2)
        self.log.add(f"[b {orange}]Starting training at: {start}")

        sw = (
            np.array(self.config["sample_weight"])
            if self.config["sample_weight"] is not None
            else None
        )

        with train_progress:
            history = model.fit(
                x,
                y,
                steps_per_epoch=self.config["steps_per_epoch"],
                epochs=self.config["EPOCHS"],
                verbose=0,
                callbacks=[self.callback],
                use_multiprocessing=True,
                workers=24,
                # validation_split=0.2,
                sample_weight=sw,
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
        self.log.add(f"Saving model at:\n {self.rnn_weights_save_path}")
        model.save(self.rnn_weights_save_path)

        self.wrap_up(model, history)

    def wrap_up(self, model, history):
        self.log.print()

        # Save a bunch of stuff
        self.save_log(self.log)
        self.save_data_to_training_folder()
        self.save_config()

        # Plot a bunch of stuff
        self.plot_weights(model)
        self.plot_training_history(history)
        try:
            self.plot_training_evaluation(model)
        except Exception:
            pass
        plt.show()

    def plot_weights(self, model):

        f, axarr = plt.subplots(
            ncols=3, nrows=len(model.layers), figsize=(16, 9)
        )

        for ax in axarr.flatten():
            ax.axis("equal")
            ax.axis("off")

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
        # Get an example batch
        y_pred, exc = None, None
        for i in range(5):
            x, y, mask, trial_params = self.eval_task.get_trial_batch()

            # predict
            try:
                y_pred = model.predict(x)
            except Exception as e:
                exc = e
            else:
                break

        if y_pred is None:
            print(f"Failed to get prediction: {exc}")
            return

        f, axarr = plt.subplots(ncols=2)
        for i, label in enumerate(["x", "y", "\\theta", "v", "\\omega"]):
            axarr[0].plot(x[0, :, i], label="true " + f"${label}$")

        for i, label in enumerate(["\\tau_{R}", "\\tau_{L}"]):
            axarr[1].plot(
                y[0, :, i], lw=2, ls="--", label="true " + f"${label}$"
            )
            axarr[1].plot(y_pred[0, :, i], label="pred " + f"${label}$")

        axarr[0].legend()
        axarr[1].legend()
        axarr[0].set(title="input", xlabel="epoch", ylabel="val")
        axarr[1].set(title="Control", xlabel="epoch", ylabel="val")
        clean_axes(f)
        save_figure(f, self.folder / "example", verbose=False)

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
