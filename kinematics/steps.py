import sys

sys.path.append("./")

from kinematics._steps import (
    get_paw_steps_times,
    get_diagonal_steps,
    step_times,
)

# TODO finish refactoring this code
# TODO check DLC
# TODO make RNN work


class Steps:
    def __init__(self, trial):
        """
            Extract step start/end times from the tracking data of a trial
        """
        self.trial = trial

    def get_steps_per_paw(self):
        """
            Get the steps for each single paw (start/end o swing phases)
        """
        self.LH_steps = get_paw_steps_times(
            self.left_hl.speed, step_speed_th, precise_th=precise_th
        )
        self.RF_steps = get_paw_steps_times(
            self.right_fl.speed, step_speed_th, precise_th=precise_th
        )

        self.RH_steps = get_paw_steps_times(
            self.right_hl.speed, step_speed_th, precise_th=precise_th
        )
        self.LF_steps = get_paw_steps_times(
            self.left_fl.speed, step_speed_th, precise_th=precise_th
        )

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
        # get steps for each paw
        self.get_steps_per_paw(precise_th)

        # Get diagonal steps (opposite pairs both moving)
        L_diagonal_steps, L_diag_data = get_diagonal_steps(
            self.LH_steps,
            self.RF_steps,
            self.trial.left_hl.speed,
            self.trial.right_fl.speed,
        )

        R_diagonal_steps, R_diag_data = get_diagonal_steps(
            self.RH_steps,
            self.LF_steps,
            self.trial.right_hl.speed,
            self.trial.left_fl.speed,
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
            LH_steps,
            RF_steps,
            LF_steps,
            RH_steps,
            R_diagonal_steps,
            L_diagonal_steps,
            diagonal_steps,
            diag_data,
            step_starts,
        )
