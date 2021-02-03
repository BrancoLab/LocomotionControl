import sys
import numpy as np

sys.path.append("./")

from kinematics._steps import (
    get_paw_steps_times,
    get_diagonal_steps,
    step_times,
)
from kinematics.fixtures import PAWS_NAMES


class Steps:
    def __init__(self, trial):
        """
            Extract step start/end times from the tracking data of a trial
        """
        self.trial = trial

    def extract_steps(
        self, step_speed_th, precise_th=12,
    ):
        """ 
            Given speed traces for the four limbs, this function returns the 
            steps the animal took, in particular it looks for 'diagonal' steps
            (when the animal is in trot) involving pairs of diagonally connected limbs

            Arguments:
                step_speed_th: float, initial threshold for step onset/offset (swing phase)
                precise_th: float, second threshold for more accurately identifying the
                    onset and offset of steps

            Returns:
                LH/LF/RH/RF_steps: step_times named tuples with onset/offset of steps
                    for each limb
                diagonal_steps: step_times named tuples with onset/offset of diagonal steps
                diag_data: dictionary with data about each diagonal step (e.g. which side, based
                    on hind limb)
                step_starts: 1d numpy array with time of each RH diagonal step, useful for plotting.

        """
        # get swing phase start/stop for each paw
        swing_phases = {
            paw: get_paw_steps_times(
                self.trial.speeds[paw], step_speed_th, precise_th=precise_th
            )
            for paw in PAWS_NAMES
        }
        self.get_steps_per_paw(precise_th)

        # Get diagonal steps (opposite pairs both moving)
        L_diagonal_steps, L_diag_data = get_diagonal_steps(
            swing_phases["left_hl"], swing_phases["right_fl"],
        )

        R_diagonal_steps, R_diag_data = get_diagonal_steps(
            swing_phases["right_hl"], swing_phases["left_fl"],
        )

        # merge R/L diagonal steps to get complete step
        _starts = np.concatenate(
            (R_diagonal_steps.starts, L_diagonal_steps.starts)
        )
        _names = np.concatenate(
            [
                [
                    "R" + str(x + 1)
                    for x in np.arange(len(R_diagonal_steps.starts))
                ],
                [
                    "L" + str(x + 1)
                    for x in np.arange(len(L_diagonal_steps.starts))
                ],
            ]
        )
        _names = _names[np.argsort(_starts)]

        pooled_starts, pooled_ends, diag_data = [], [], {}
        count = 0
        for n in _names:
            side, idx = n[0], int(n[1])
            if side == "R":
                pooled_starts.append(R_diagonal_steps.starts[idx - 1])
                pooled_ends.append(R_diagonal_steps.ends[idx - 1])
                data = R_diag_data[idx - 1]
                data["paws"] = ("right_hl", "left_fl")
            else:
                pooled_starts.append(L_diagonal_steps.starts[idx - 1])
                pooled_ends.append(L_diagonal_steps.ends[idx - 1])
                data = L_diag_data[idx - 1]
                data["paws"] = ("left_hl", "right_fl")

            data["side"] = side
            diag_data[count] = data
            count += 1
        diagonal_steps = step_times(pooled_starts, pooled_ends)

        # step starts
        step_starts = (
            np.array(R_diagonal_steps.starts) + self.start
        )  # to mark the start of each L-R step sequence

        return (
            swing_phases,
            R_diagonal_steps,
            L_diagonal_steps,
            diagonal_steps,
            diag_data,
            step_starts,
        )
