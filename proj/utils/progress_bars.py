from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
)
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
