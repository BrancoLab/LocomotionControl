import numpy as np
from collections import namedtuple
import pandas as pd
from scipy.stats import pearsonr

from rich.table import Table
from rich import print
from rich.box import SIMPLE_HEAD

from myterial import blue, salmon, pink_light

from fcutils.maths.signals import derivative

# ----------------------------------- misc ----------------------------------- #


def print_steps_summary(summary):
    """
        Prints a Rich table with an overview of a set of steps
        Summary is a dict or df with steps data.
    """

    def foot(x):
        return f"{np.mean(x):.2f}"  # " +/- {np.std(x):.1f}"

    if isinstance(summary, dict):
        summary = pd.DataFrame(summary)

    main_tb = Table(
        header_style="bold green", show_lines=False, expand=False, box=None,
    )
    main_tb.add_column("LEFT", justify="center")
    main_tb.add_column("", justify="center", width=10)
    main_tb.add_column("RIGHT", justify="center")
    L, R = summary.loc[summary.side == "L"], summary.loc[summary.side == "R"]

    subtables = []
    for summary in (L, R):
        tb = Table(
            header_style="bold green",
            show_lines=True,
            expand=False,
            box=SIMPLE_HEAD,
            show_footer=True,
            footer_style=f"{pink_light} bold",
        )
        tb.add_column("#", style="dim", footer=str(len(summary)) + " steps")
        tb.add_column("side",)
        tb.add_column("start", justify="center", width=4)
        tb.add_column("end", justify="center", width=4)
        tb.add_column(
            "dur.",
            justify="center",
            footer=foot(summary.end - summary.start),
            width=4,
        )
        tb.add_column(
            "stride delta", justify="right", footer=foot(summary.stride_delta)
        )
        tb.add_column(
            "angle delta", justify="right", footer=foot(summary.angle_delta)
        )
        tb.add_column("Correlation", justify="center")

        for i, step in summary.iterrows():
            tb.add_row(
                str(step["number"]),
                step["side"],
                str(step["start"]),
                str(step["end"]),
                str(step["end"] - step["start"]),
                f"{step['stride_delta']:.3f}",
                f"{step['angle_delta']:.3f}",
                f"{step['pearsonr']:.3f}",
                style=blue if step["side"] == "R" else salmon,
            )
        subtables.append(tb)

    main_tb.add_row(subtables[0], " ", subtables[1])
    print("\n", main_tb)


# ---------------------------------------------------------------------------- #
#                                  steps times                                 #
# ---------------------------------------------------------------------------- #

step_times = namedtuple("step_times", "starts, ends")


def get_paw_steps_times(speed, step_speed_th, precise_th=12):
    """
        Given the speed of a paw, find when the swing
        phase starts and ends.

        First it finds the times where the paw speed 
        was >= stop speed th, then it finds the onset/offset
        more precisly by looking for when the speed
        went above or below precise_th cm/s
    """
    # get approximate times
    is_swing = np.zeros_like(speed)
    is_swing[speed > step_speed_th] = 1

    first_zero = np.where(is_swing == 0)[0][0]
    is_swing[:first_zero] = 0  # make sure that is starts with swing phase OFF

    starts = np.where(derivative(is_swing) > 0)[0]
    ends = np.where(derivative(is_swing) < 0)[0]

    # Get precise times
    precise_starts, precise_ends = [], []
    for s, e in zip(starts, ends):
        try:
            precise_starts.append(np.where(speed[:s] <= precise_th)[0][-1] + 1)
        except IndexError:
            precise_starts.append(s)

        try:
            precise_ends.append(np.where(speed[e:] <= precise_th)[0][0] + e)
        except IndexError:
            precise_ends.append(e)
    return step_times(precise_starts, precise_ends)


def get_diagonal_steps(hind, fore, hind_speed, fore_speed):
    """
        Given the start/end times of the swing 
        phases for a hind paw and a (diagonally
        opposed) fore paw, get the  start/end
        time of step. It assumes that mouse
        is engaged in diagonal stepping gait.

        A step starts when the first of H/F starts
        moving and ends when the last stops moving

        Returns a step_times tuple of steps times
        and a dictionary of dicts that say when each paw starts/stops
        for  each step

        Arguments: 
            hind/fore: step_times namedtuples with start/end of steps
            hind/fore_speed: 1d numpy arrays with paw speed
    """
    # get an arr that is 1 when either is stepping
    last = max(hind.ends[-1], fore.ends[-1])
    arr = np.zeros(last + 1)

    for paw in (hind, fore):
        for s, e in zip(paw.starts, paw.ends):
            arr[s:e] = 1

    # get starts and ends
    starts = np.where(derivative(arr) > 0)[0]
    ends = np.where(derivative(arr) < 0)[0]
    if arr[0] == 1:
        starts = np.concatenate([[0], starts])

    if arr[-1] == 1:
        ends = np.concatenate([ends, [len(arr)]])

    if ends[0] < starts[0] or starts[-1] > ends[-1]:
        raise ValueError("Something went wrong while getting starts and ends")

    # now create data dict
    data = {}
    good_starts, good_ends = [], []
    count = 0
    for s, e in zip(starts, ends):
        # get  starts and stops
        try:
            data[count] = dict(
                start=s,
                end=e,
                fore_start=[start for start in fore.starts if start >= s][0],
                fore_end=[end for end in fore.ends if end <= e][-1],
                hind_start=[start for start in hind.starts if start >= s][0],
                hind_end=[end for end in hind.ends if end <= e][-1],
                pearsonr=pearsonr(fore_speed[s:e], hind_speed[s:e])[0],
            )
        except Exception:
            continue
        else:
            good_starts.append(s)
            good_ends.append(e)

        if data[count]["fore_start"] < s or data[count]["hind_start"] < s:
            raise ValueError(f"Limb start before step start: {data[count]}")

        if data[count]["fore_end"] > e or data[count]["hind_end"] > e:
            raise ValueError(f"Limb end after step end: {data[count]}")

        count += 1

    if len(good_starts) != len(good_ends):
        raise ValueError(
            f"Different number of starts and stops: {len(good_starts)}-{len(good_ends)}"
        )
    if len(good_starts) != len(data.keys()):
        raise ValueError(
            f"Wrong number of entries in data dictionary: {len(data.keys())} instead of {len(good_starts)}\n{data}"
        )

    return step_times(good_starts, good_ends), data
