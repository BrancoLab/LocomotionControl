from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
)
from rich.text import Text
import GPUtil as GPU


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
            return Text(" ")
        else:
            return Text(f"{task.speed:.3f} steps/s")


class LossColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        try:
            return Text(f"loss: {task.loss:.3e}")
        except AttributeError:
            try:
                return Text(f"loss: {task.fields['loss']:.3e}")
            except AttributeError:
                print("failed")
            return "no loss"


class GPUColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        gpu = get_gpu()
        if gpu is None:
            return Text("no gpu")
        else:
            return Text(
                f"Used GPU mem: {int(gpu.memoryUsed)}/{int(gpu.memoryTotal)} MB"
            )


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
