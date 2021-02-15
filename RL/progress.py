from rich.layout import Layout
from rich.panel import Panel
from rich.align import Align
from rich import box
from rich.table import Table
import numpy as np
import inspect
from pathlib import Path
import termplotlib as tpl

from myterial import (
    pink,
    orange,
    pink_light,
    indigo,
    grey,
    blue_light,
    green_light,
    green,
    salmon,
)

from fcutils.progress import progess_with_description as progress
from fcutils.time import timestamp

from control import model


def fmt(obj):
    if isinstance(obj, tuple):
        try:
            return {k: round(v, 2) for k, v in obj._asdict().items()}
        except Exception:
            return list(round(item, 2) for item in obj)
    elif isinstance(obj, np.ndarray):
        return [round(v, 2) for v in obj]
    elif isinstance(obj, dict):
        return {k: round(v, 2) for k, v in obj.items()}
    elif isinstance(obj, float):
        return str(round(obj, 2))
    else:
        return ""


class Trajectory:
    trajectory = None

    def __rich_console__(self, *args):
        if self.trajectory is not None:
            yield ""
        else:
            yield ""


class RewardPlot:
    scores = []

    def __rich_console__(self, *args):
        x = np.arange(len(self.scores))

        fig = tpl.figure()
        fig.plot(x, self.scores, label="r", width=100, height=15)
        yield fig.get_string()


class StackInfo:
    prev_scopes = []

    def __rich_console__(self, *args, **kwargs):
        yield ""
        scopes = [
            s
            for s in inspect.stack()
            if "rich" not in s.filename
            and "python" not in s.filename
            and "rich" not in s.function
        ]
        if not scopes:
            scopes = self.prev_scopes

        for scope in scopes[::-1]:
            frame = scope.frame
            path = Path(frame.f_code.co_filename)
            yield f"Running: [{green}]{path.parent.name}/{path.name}[/{green}] [dim](line: {frame.f_lineno})[/dim] - [bold {green_light}]{scope.function}"
        self.prev_scopes = scopes


class State:
    tb = ""
    controls = ""
    prev_v = 0
    prev_omega = 0

    def update(self, agent, world, controls):
        self.tb = Table(box=None, header_style=f"bold {blue_light}")
        self.tb.add_column("Name")
        for k in agent.curr_x._asdict().keys():
            self.tb.add_column(k, style="white")

        self.tb.add_row(
            f"[{orange}]World",
            *[f"{v:.1f}" for v in world.get_next_waypint()._asdict().values()],
        )
        self.tb.add_row(
            f"[{orange}]Agent",
            *[f"{v:.1f}" for v in agent.curr_x._asdict().values()],
        )
        try:
            self.tb.add_row(
                f"[{orange}]Dxdt",
                *[f"{v:.1f}" for v in agent.curr_dxdt._asdict().values()],
            )
        except AttributeError:
            pass

        # controls
        self.controls = Table(box=None, header_style=f"bold {blue_light}")
        self.controls.add_column("Var", justify="right", style=orange)
        self.controls.add_column("Value", style=green_light, justify="left")

        for k, v in model.control(*controls)._asdict().items():
            self.controls.add_row(k, str(fmt(v)))
        self.controls.add_row("", "")

        # vdot omegadot
        self.controls.add_row("v_dot", fmt(agent.curr_x.v - self.prev_v))
        self.controls.add_row(
            "omega_dot", fmt(agent.curr_x.omega - self.prev_omega)
        )

        self.prev_v = agent.curr_x.v
        self.prev_omega = agent.curr_x.omega

        # wheel speeds
        try:
            l, r = agent.get_wheel_velocities()
        except AttributeError:
            pass
        else:
            self.controls.add_row("", "")
            self.controls.add_row("phi_R", fmt(r))
            self.controls.add_row("phi_L", fmt(l))

    def __rich_console__(self, *args, **kwargs):
        yield f"[b {pink}]States"
        yield self.tb
        yield ""
        yield f"[b {pink}]Controls & physics"
        yield self.controls


class Variables:
    max_score = 0
    traj_len = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self.itern = 0
        self.reward = 0
        self.inputs = 0
        self.traj_distance = 0
        self.traj_idx = 0
        self.score = 0
        self.memory_len = 0

    def __rich_console__(self, *args, **kwargs):
        yield f"[white]Iteration number: [b {orange}]{self.itern}"
        yield f"[white]Memory length: [b {orange}]{self.memory_len}"
        yield f"[white]Trajectory idx: [b {orange}]{self.traj_idx}/{self.traj_len}"
        yield f"[white]Reward: [b {orange}]{self.reward:.3f}"
        yield f"[white]Score: [b {orange}]{self.score:.3f}"
        yield f"[white]MAX score: [b {orange}]{self.max_score:.3f}"
        yield f"[white]Inputs: [b {orange}]{fmt(self.inputs)}"


class Progress:
    max_log_rows = 12

    def __init__(self, agent, variables, state, reward_plot):
        self.progress = progress
        self.layout = self.make_layout(agent, variables, state, reward_plot)

    def make_header(self):
        title_panel = Panel(
            Align.center(f"[b {pink}]Training A2C driver", vertical="middle"),
            border_style=f"{pink_light}",
        )

        info_panel = Panel(
            f"Author: [b]Federico Claudi[/b]\nDate: [b]{timestamp()}",
            padding=(0, 3),
            box=box.SIMPLE_HEAD,
        )
        return title_panel, info_panel

    def footer(self):
        return Layout(
            Panel(self.progress, border_style=orange),
            name="footer",
            size=4,
            ratio=1,
        )

    def make_logger(self):
        self.logger = Table(
            expand=True,
            header_style=f"b white",
            show_lines=True,
            style=grey,
            box=box.SIMPLE_HEAD,
        )
        self.logger.add_column(
            header="time", justify="left", style="dim", width=8, max_width=8
        )
        self.logger.add_column(
            header="Info",
            justify="right",
            style="b yellow",
            width=5,
            max_width=10,
        )
        self.logger.add_column(
            header="details", justify="left", style="dim", min_width=32
        )

        return self.logger

    def cut_logger(self):
        # make a new table with fewer rows
        for col in self.logger.columns:
            col._cells = col._cells[-self.max_log_rows :]

    def log(self, info, details):
        if len(self.logger.rows) > self.max_log_rows:
            self.cut_logger()
        self.logger.add_row(timestamp(just_time=True), info, details)

    def make_layout(self, agent, variables, state, reward_plot):
        layout = Layout()
        self.make_logger()

        layout.split(Layout(name="head", ratio=2, size=6))
        layout.split(Layout(name="body", ratio=5, direction="horizontal"))
        layout.split(Layout(name="foot", ratio=1, size=4))

        # make header
        title_panel, info_panel = self.make_header()
        layout["head"].split(Layout(name="H-l"), direction="horizontal")
        layout["head"]["H-l"].update(title_panel)
        layout["head"].split(Layout(name="H-r"), direction="horizontal")
        layout["head"]["H-r"].update(info_panel)

        # make body
        layout["body"].split(Layout(name="bl"))
        layout["body"].split(Layout(name="br", ratio=2))

        layout["body"]["br"].split(Layout(name="brt"))
        layout["body"]["br"].split(Layout(name="brb", ratio=2))

        layout["body"]["br"]["brt"].update(
            Panel(
                reward_plot,
                border_style=salmon,
                box=box.SIMPLE,
                title="reward history",
            )
        )
        layout["body"]["br"]["brb"].update(
            Panel(self.make_logger(), style=indigo, title="LOG")
        )

        # make body left
        layout["body"]["bl"].split(Layout(name="blt"))
        layout["body"]["bl"].split(Layout(name="blm", ratio=2))
        layout["body"]["bl"].split(Layout(name="blb"))

        layout["body"]["bl"]["blt"].update(
            Panel(variables, style=orange, padding=(1, 3), title="Variables")
        )
        layout["body"]["bl"]["blm"].update(
            Panel(state, style=green, title="Physics")
        )
        layout["body"]["bl"]["blb"].update(
            Panel(StackInfo(), title="Call stack", box=box.SIMPLE_HEAD)
        )

        # make footer
        layout["foot"].update(self.footer())
        return layout

    def add_task(self, *args, **kwargs):
        return progress.add_task(*args, **kwargs)

    def update_task(self, tid, *args, **kwargs):
        self.progress.update(tid, *args, **kwargs)

    def remove_task(self, tid):
        self.progress.remove_task(tid)
