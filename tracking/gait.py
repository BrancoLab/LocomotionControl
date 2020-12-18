import numpy as np
from collections import namedtuple

from fcutils.maths.utils import derivative
from fcutils.maths.geometry import calc_distance_between_points_2d


# ----------------------------------- misc ----------------------------------- #


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
    for n, (start, end) in enumerate(zip(leading.starts, leading.ends)):
        # get the next trailing step
        trailing_start = [s for s in trailing.starts if s > start]
        if not trailing_start:
            break

        # get the end for the next trailing step
        trailing_end = [e for e in trailing.ends if e > trailing_start[0]]
        if trailing_end:
            starts.append(start)
            ends.append(trailing_end[0])

            data[n] = dict(
                leading_start=start,
                leading_end=end,
                trailing_start=trailing_start[0],
                trailing_end=trailing_end[0],
            )

    return step_times(starts, ends), first, data


def get_diag_steps(hind, fore):
    """
        Given the start/end times of the swing 
        phases for a hind paw and a (diagonally
        opposed) fore paw, get the  start/end
        time of step.

        A step starts when the first of H/F starts
        moving and ends when the last stops moving

        Returns a step_times tuple of steps times
    """

    return
