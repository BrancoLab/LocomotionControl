from rich.layout import Layout
from rich.panel import Panel
from rich.align import Align
from rich import box
from rich.table import Table
import numpy as np

from myterial import (
    pink,
    orange,
    pink_light,
    indigo,
    grey,
    amber_light,
    blue_light,
)

from fcutils.progress import progess_with_description as progress
from fcutils.time import timestamp


class State:
    tb = ""

    def update(self, agent, world):
        self.tb = Table(box=None, header_style=f"bold {blue_light}")
        self.tb.add_column("Name")
        for k in agent.curr_x._asdict().keys():
            self.tb.add_column(k)

        self.tb.add_row(
            f"[{orange}]World",
            *[f"{v:.1f}" for v in world.get_next_waypint()._asdict().values()],
        )
        self.tb.add_row(
            f"[{orange}]Agent",
            *[f"{v:.1f}" for v in agent.curr_x._asdict().values()],
        )

    def __rich_console__(self, *args, **kwargs):
        yield self.tb


class Variables:
    itern = 0
    reward = 0
    action = None
    state = 0
    traj_distance = 0
    traj_idx = 0
    error = (0, 0)

    def reset(self):
        self.itern = 0
        self.reward = 0
        self.action = None
        self.state = 0
        self.traj_distance = 0
        self.traj_idx = 0
        self.error = (0, 0)

    def fmt(self, obj):
        if isinstance(obj, tuple):
            try:
                return {k: round(v, 2) for k, v in obj._asdict().items()}
            except Exception:
                return list(round(item, 2) for item in obj)
        elif isinstance(obj, np.ndarray):
            return [round(v, 2) for v in obj]
        else:
            return ""

    def __rich_console__(self, *args, **kwargs):
        yield f"[{amber_light}]Iteration number: [b {orange}]{self.itern}"
        yield f"[{amber_light}]Reward: [b {orange}]{self.reward:.3f}"
        yield f"[{amber_light}]Error: [b {orange}]{self.fmt(self.error)}"
        yield f"[{amber_light}]Action: [b {orange}]{self.fmt(self.action.ravel())}" if self.action is not None else ""
        yield f"[{amber_light}]State: [b {orange}]{self.fmt(self.state)}"
        yield f"[{amber_light}]Trajectory distance: [b {orange}]{self.traj_distance:.3f}"
        yield f"[{amber_light}]Trajectory idx: [b {orange}]{self.traj_idx}"


class Progress:
    max_log_rows = 10

    def __init__(self, agent, variables, state):
        self.progress = progress
        self.layout = self.make_layout(agent, variables, state)

    def header(self):
        title_panel = Panel(
            Align.center(f"[b {pink}]Training A2C driver", vertical="middle"),
            border_style=f"{pink_light}",
        )

        info_panel = Panel(
            f"Author: [b]Federico Claudi[/b]\nDate: [b]{timestamp()}",
            padding=(0, 3),
            box=box.SIMPLE_HEAD,
        )
        return Layout(title_panel), Layout(info_panel)

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

    def make_layout(self, agent, variables, state):
        layout = Layout()
        header, pane2, pane3 = layout.split(
            Layout(size=4),
            Layout(ratio=3),
            self.footer(),
            direction="vertical",
        )
        header.split(*self.header(), direction="horizontal")

        agent_side, self.logger_layout = pane2.split(
            Layout(),
            Layout(Panel(self.make_logger(), style=indigo, title="LOG")),
            direction="horizontal",
        )

        actor_s, critic_s = agent_side.split(
            Layout(
                Panel(
                    variables, style=orange, padding=(1, 3), title="Variables"
                )
            ),
            Layout(Panel(state)),
        )
        return layout

    def add_task(self, *args, **kwargs):
        return progress.add_task(*args, **kwargs)

    def update_task(self, tid, *args, **kwargs):
        self.progress.update(tid, *args, **kwargs)

    def remove_task(self, tid):
        self.progress.remove_task(tid)
