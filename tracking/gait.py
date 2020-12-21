import numpy as np
from collections import namedtuple
import pandas as pd

from rich.table import Table
from rich import print
from rich.box import SIMPLE_HEAD

from myterial import blue, salmon

from fcutils.maths.utils import derivative
from fcutils.maths.geometry import calc_distance_between_points_2d


# ----------------------------------- misc ----------------------------------- #
def print_steps_summary(summary):
    """
        Summary is a dict or df with steps data
    """

    def foot(x):
        return f"{np.mean(x):.3f} +/ {np.std(x):.3f}"

    if isinstance(summary, dict):
        summary = pd.DataFrame(summary)

    tb = Table(
        header_style="bold green",
        show_lines=True,
        expand=False,
        box=SIMPLE_HEAD,
        show_footer=True,
        footer_style="magenta",
    )
    tb.add_column("#", style="dim", footer=str(len(summary)))
    tb.add_column("side",)
    tb.add_column("start", justify="center", width=4)
    tb.add_column("end", justify="center", width=4)
    tb.add_column(
        "dur.", justify="center", footer=foot(summary.end - summary.start)
    )
    tb.add_column(
        "stride delta", justify="right", footer=foot(summary.stride_delta)
    )
    tb.add_column(
        "angle delta", justify="right", footer=foot(summary.angle_delta)
    )

    for i, step in summary.iterrows():
        tb.add_row(
            str(step["number"]),
            step["side"],
            str(step["start"]),
            str(step["end"]),
            str(step["end"] - step["start"]),
            f"{step['stride_delta']:.3f}",
            f"{step['angle_delta']:.3f}",
            style=blue if step["side"] == "R" else salmon,
        )

    print("\n", tb)


def stride_from_speed(speed, start, end):
    """
        Given a paw's speed trace and the
        start and end of the swing phase of a  step
        returns the length of the stride
    """
    return np.cumsum(speed[start:end])[-1]


def stride_from_position(x, y, start, end):
    """
        Given the xy tracking of a paw and the
        start and end of the swing phase of a  step
        returns the length of the stride
    """
    p1 = (x[start], y[start])
    p2 = (x[end], y[end])
    return calc_distance_between_points_2d(p1, p2)


# ---------------------------------------------------------------------------- #
#                                  steps times                                 #
# ---------------------------------------------------------------------------- #

step_times = namedtuple("step_times", "starts, ends")


def get_paw_steps_times(speed, step_speed_th):
    """
        Given the speed of a paw, find when the swing
        phase starts and ends
    """
    is_swing = np.zeros_like(speed)
    is_swing[speed > step_speed_th] = 1

    first_zero = np.where(is_swing == 0)[0][0]
    is_swing[:first_zero] = 0  # make sure that is starts with swing phase OFF

    starts = np.where(derivative(is_swing) > 0)[0]
    ends = np.where(derivative(is_swing) < 0)[0]
    return step_times(starts, ends)


def get_diagonal_steps_pairs(left, right):
    """
        Given the start and end of the swing phase
        of a left and right paw (e.g. LH, RH), get 
        the start/end time of each diagonal step.

        A diagonal step is defined by 4 events:
            - START: onset of left's swing
            - end of left's swing
            - start of right's swing
            - END: end of right's swing

        left, right should be instances of 'step_times' named tuple for
        each paw's swing phases(from get_paw_steps_times)

        Returns:
            - start of diag steps (list of frames)
            - end of diag steps (list of frames)
            - which side was first ('left' or 'right')
            - data: a dict where for each step you have the start and end time of each side
    """
    if left.starts[0] < right.starts[0]:
        # left paw starts
        first = "left"
        leading, trailing = left, right
    else:
        first = "right"
        leading, trailing = right, left

    starts, ends, data = [], [], {}
    count = 0
    for start, end in zip(leading.starts, leading.ends):
        # get the next trailing step
        trailing_start = [s for s in trailing.starts if s > start]
        if not trailing_start:
            break

        # get the end for the next trailing step
        trailing_end = [e for e in trailing.ends if e > trailing_start[0]]
        if trailing_end:
            starts.append(start)
            ends.append(trailing_end[0])

            data[count] = dict(
                leading_start=start,
                leading_end=end,
                trailing_start=trailing_start[0],
                trailing_end=trailing_end[0],
            )
            count += 1

    return step_times(starts, ends), first, data


def get_diagonal_steps(hind, fore):
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
    """
    # get an arr that is 1 when either is stepping
    last = max(hind.ends[-1], fore.ends[-3])
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
