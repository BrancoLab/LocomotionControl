import sys

from fcutils.maths.coordinates import cart2pol
from fcutils.maths.array import unwrap
from fcutils.maths.signals import derivative

sys.path.append("./")
from rnn.dataset._dataset import Preprocessing, Dataset


is_win = sys.platform == "win32"

# ---------------------------------------------------------------------------- #
#                             cartesian coordinates                            #
# ---------------------------------------------------------------------------- #


class PredictTauFromDeltaXYT(Dataset, Preprocessing):
    description = """
        Predict torques (tau right/left) from 
        the  trajectory delta: difference betwen current state
        and next goal in the control simulation.

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """
    polar = False
    name = "dataset_predict_tau_from_deltaXYT"
    inputs_names = ("x", "y", "theta")
    outputs_names = ("tau_R", "tau_L")
    n_frames_ahead = -1

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history):
        dx = history.goal_x - history.x
        dy = history.goal_y - history.y
        dtheta = history.goal_theta - history.theta

        return dx, dy, dtheta

    def get_outputs(self, history):
        return history.tau_r, history.tau_l


class PredictPNNFromDeltaXYT(Dataset, Preprocessing):
    description = """
        Predict neural controls (P, N_L, N_R) from 
        the  trajectory delta: difference betwen current state
        and next goal in the control simulation.

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """
    polar = False
    name = "dataset_predict_PNN_from_deltaXYT"
    inputs_names = ("x", "y", "theta")
    outputs_names = ("P", "N_R", "N_L")
    n_frames_ahead = -1

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

    def get_inputs(self, trajectory, history):
        dx = history.goal_x - history.x
        dy = history.goal_y - history.y
        dtheta = history.goal_theta - history.theta

        return dx, dy, dtheta

    def get_outputs(self, history):
        return history.P, history.N_r, history.N_l


# ---------------------------------------------------------------------------- #
#                               POLAR COORDINATES                              #
# ---------------------------------------------------------------------------- #


class PredictPNNFromRPsyVO(Dataset, Preprocessing):
    description = """
        Predict neural controls (P, N_L, N_R) from 
        the distance to the next state along the trajectory in
        polar coordinates and the current linear and angular velocity.
        The 'next' state is defined as the state in the trajectory 100 ms in
        the future compared to the current state.

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """
    polar = True

    name = "dataset_predict_PNN_from_RPsyVO"
    inputs_names = ("r", "psy", "v", "omega")
    outputs_names = ("P", "N_R", "N_L")

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

        self.n_frames_ahead = int(
            500 / 5
        )  # 0.005s is the dt of the simulations

    def get_inputs(self, trajectory, history):
        # Get current states and states 500 ms in the future
        future_states = history.iloc[self.n_frames_ahead :].reset_index()
        current_states = history.iloc[: -self.n_frames_ahead].reset_index()

        # take the distance to the future states at each time point
        dx = future_states.x - current_states.x
        dy = future_states.y - current_states.y

        # convert to polar coordinates
        r, psy = cart2pol(dx, dy)
        psy = unwrap(psy)

        return r, psy, current_states.v.values, current_states.omega.values

    def get_outputs(self, history):
        return (
            history.P.iloc[: -self.n_frames_ahead],
            history.N_r.iloc[: -self.n_frames_ahead],
            history.N_l.iloc[: -self.n_frames_ahead],
        )


class PredictTauFromRPsyVO(Dataset, Preprocessing):
    description = """
        Predict neural torques (tau right/left) from 
        the distance to the next state along the trajectory in
        polar coordinates and the current linear and angular velocity.
        The 'next' state is defined as the state in the trajectory 500 ms in
        the future compared to the current state.

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """
    polar = True

    name = "dataset_predict_Tau_from_RPsyVO"
    inputs_names = ("r", "psy", "v", "omega")
    outputs_names = ("tau_R", "tau_L")

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

        self.n_frames_ahead = int(
            500 / 5
        )  # 0.005s is the dt of the simulations

    def get_inputs(self, trajectory, history):
        # Get current states and states 500 ms in the future
        future_states = history.iloc[self.n_frames_ahead :].reset_index()
        current_states = history.iloc[: -self.n_frames_ahead].reset_index()

        # take the distance to the future states at each time point
        dx = future_states.x - current_states.x
        dy = future_states.y - current_states.y

        # convert to polar coordinates
        r, psy = cart2pol(dx, dy)
        psy = unwrap(psy)

        return r, psy, current_states.v.values, current_states.omega.values

    def get_outputs(self, history):
        return (
            history.P.iloc[: -self.n_frames_ahead],
            history.N_r.iloc[: -self.n_frames_ahead],
            history.N_l.iloc[: -self.n_frames_ahead],
        )


class PredictWheelsFromRPsyVO(Dataset, Preprocessing):
    description = """
        Predict the speed of the left and right wheels Point of Contact based on current velocities and 
        the polar distance to the next goal.
        The 'next' state is defined as the state in the trajectory 500 ms in
        the future compared to the current state.

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """
    polar = True

    name = "dataset_predict_wheels_from_RPsyVO"
    inputs_names = ("r", "psy", "v", "omega")
    outputs_names = ("phidot_r", "phidot_l")

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

        self.n_frames_ahead = int(
            500 / 5
        )  # 0.005s is the dt of the simulations

    def get_inputs(self, trajectory, history):
        # Get current states and states 500 ms in the future
        future_states = history.iloc[self.n_frames_ahead :].reset_index()
        current_states = history.iloc[: -self.n_frames_ahead].reset_index()

        # take the distance to the future states at each time point
        dx = future_states.x - current_states.x
        dy = future_states.y - current_states.y

        # convert to polar coordinates
        r, psy = cart2pol(dx, dy)
        psy = unwrap(psy)

        return r, psy, current_states.v.values, current_states.omega.values

    def get_outputs(self, history):
        return history.phidot_r, history.phidot_l


class PredictAccelerationsFromRPsyVO(Dataset, Preprocessing):
    description = """
        Predict the linear and angular accelerations based on current velocities and 
        the polar distance to the next goal.
        The 'next' state is defined as the state in the trajectory 500 ms in
        the future compared to the current state.

        Data are normalized in range (-1, 1) with a MinMaxScaler for each
        variable independently.
    """
    polar = True

    name = "dataset_predict_accelerations_from_RPsyVO"
    inputs_names = ("r", "psy", "v", "omega")
    outputs_names = ("vdot", "omegadot")

    def __init__(self, *args, truncate_at=None, **kwargs):
        Preprocessing.__init__(self, truncate_at=truncate_at, **kwargs)
        Dataset.__init__(self, *args, **kwargs)

        self.n_frames_ahead = int(
            500 / 5
        )  # 0.005s is the dt of the simulations

    def get_inputs(self, trajectory, history):
        # Get current states and states 500 ms in the future
        future_states = history.iloc[self.n_frames_ahead :].reset_index()
        current_states = history.iloc[: -self.n_frames_ahead].reset_index()

        # take the distance to the future states at each time point
        dx = future_states.x - current_states.x
        dy = future_states.y - current_states.y

        # convert to polar coordinates
        r, psy = cart2pol(dx, dy)
        psy = unwrap(psy)

        return r, psy, current_states.v.values, current_states.omega.values

    def get_outputs(self, history):
        vdot = derivative(history.v.iloc[: -self.n_frames_ahead])
        vdot[0] = vdot[1]

        omegadot = derivative(history.omega.iloc[: -self.n_frames_ahead])
        omegadot[0] = omegadot[1]
        return vdot, omegadot


if __name__ == "__main__":
    datasets = (
        # PredictTauFromDeltaXYT,
        # PredictPNNFromDeltaXYT,
        PredictPNNFromRPsyVO,
        PredictTauFromRPsyVO,
        PredictAccelerationsFromRPsyVO,
        # PredictWheelsFromRPsyVO,
    )

    for dataset in datasets:
        dataset().make()
        dataset().plot_random()
        # dataset().plot_durations()
