from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
)
from tensorflow import keras
from tensorflow.keras import backend as K
import GPUtil as GPU
from pyinspect._colors import orange, mocassin


def get_gpu():
    GPUs = GPU.getGPUs()
    if GPUs:
        return GPUs[0]
    else:
        return None


class SpeedColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        if task.speed is None:
            return " "
        else:
            return f"{task.speed:.1f} steps/s"


class LossColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        try:
            return (
                f"[{mocassin}]loss: [bold {orange}]{task.fields['loss']:.5f}"
            )
        except AttributeError:
            return "no loss"


class LearningRateColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        try:
            return f"[{mocassin}]lr: [bold {orange}]{task.fields['lr']:.4f}"
        except AttributeError:
            return "no lr"


class GPUColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        gpu = get_gpu()
        if gpu is None:
            return "no gpu"
        else:
            return f"Used GPU mem: {int(gpu.memoryUsed)}/{int(gpu.memoryTotal)} MB"


train_progress = Progress(
    TextColumn("[bold magenta]Step {task.completed}/{task.total}"),
    SpeedColumn(),
    "•",
    "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    "•",
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeRemainingColumn(),
    "•",
    LossColumn(),
    LearningRateColumn(),
    "•",
    GPUColumn(),
)


progress = Progress(
    TextColumn("[bold magenta]Step {task.completed}/{task.total}"),
    "•",
    SpeedColumn(),
    # "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    "•",
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeRemainingColumn(),
)


# -------------------------- keras custom traceback -------------------------- #


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
        self.lr = K.eval(self.lr_schedule(epoch))
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
