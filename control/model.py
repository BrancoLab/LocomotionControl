from collections import namedtuple
import numpy as np

from .dynamics import ModelDynamics
from .config import dt, MOUSE
from .utils import merge


# define some useful named tuples
control = namedtuple("control", "tau_r, tau_l")
state = namedtuple("state", "x, y, theta, v, omega")
goal = namedtuple("state", "goal_x, goal_y, goal_theta, goal_v, goal_omega")
_dxdt = namedtuple("dxdt", "x_dot, y_dot, theta_dot, v_dot, omega_dot")
wheel_state = namedtuple("wheel_state", "nudot_right, nudot_left")


class Model(ModelDynamics):
    """
        The model class uses the model's dynamics to step the simulation
        forward given a control input and the model's parameters

        States that the model class keeps track of:
            curr_x
            curr_wheel_state
            curr_goal
            control
            curr_dxdt
    """

    def __init__(self):
        ModelDynamics.__init__(self)

    def initialize(self, trajectory):
        # start th emodel at the start of the trajectory
        self.curr_x = state(
            trajectory[0, 0],
            trajectory[0, 1],
            trajectory[0, 2],
            trajectory[0, 3],
            trajectory[0, 4],
        )

    def step(self, u, curr_goal):
        # prep some variables
        self.curr_x = state(*self.curr_x)
        self.curr_goal = goal(*curr_goal)
        u = control(*np.array(u))

        variables = merge(u, self.curr_x, MOUSE)
        inputs = [variables[a] for a in self._M_args]

        # Compute wheel velocities
        w = self.calc_wheels_ang_vels(
            variables["L"], variables["R"], self.curr_x.v, self.curr_x.omega,
        )
        self.curr_wheel_state = wheel_state(*w.ravel())

        # Update history
        self.curr_control = u

        # Compute dxdt
        dxdt = self.calc_dqdt(*inputs).ravel()
        self.curr_dxdt = _dxdt(*dxdt)

        # Step
        next_x = np.array(self.curr_x) + dxdt * dt
        self.curr_x = state(*next_x)

    # ------------------------------ Control related ----------------------------- #

    def _fake_step(self, x, u):
        """
            Simulate a step fiven a state and a control
        """
        x = state(*x)
        u = control(*u)

        # Compute dxdt
        variables = merge(u, x, MOUSE)
        inputs = [variables[a] for a in self._M_args]
        dxdt = self.calc_dqdt(*inputs).ravel()

        if np.any(np.isnan(dxdt)) or np.any(np.isinf(dxdt)):
            # raise ValueError('Nans in dxdt')
            print("nans in dxdt during fake step")

        # Step
        next_x = np.array(x) + dxdt * dt
        return next_x

    def predict_trajectory(self, curr_x, us):
        """
            Compute the trajectory for N steps given a
            state and a (series of) control(s)
        """
        if len(us.shape) == 3:
            pred_len = us.shape[1]
            us = us.reshape((pred_len, -1))
            expand = True
        else:
            expand = False

        # get size
        pred_len = us.shape[0]

        # initialze
        x = curr_x  # (3,)
        pred_xs = curr_x[np.newaxis, :]

        for t in range(pred_len):
            next_x = self._fake_step(x, us[t])
            # update
            pred_xs = np.concatenate((pred_xs, next_x[np.newaxis, :]), axis=0)
            x = next_x

        if expand:
            pred_xs = pred_xs[np.newaxis, :, :]
            # pred_xs = np.transpose(pred_xs, (1, 0, 2))
        return pred_xs

    def calc_gradient(self, xs, us, wrt="x"):
        """
            Compute the models gradient wrt state or control,
            used to compute controls
        """

        # prep some variables
        theta = xs[:, 2]
        v = xs[:, 3]
        omega = xs[:, 4]

        L = MOUSE["L"]
        R = MOUSE["R"]
        m = MOUSE["m"]
        m_w = MOUSE["m_w"]
        d = MOUSE["d"]

        (_, state_size) = xs.shape
        (pred_len, input_size) = us.shape

        if wrt == "x":
            f = np.zeros((pred_len, state_size, state_size))
            for i in range(pred_len):
                f[i, :, :] = self.calc_model_jacobian_state(
                    theta[i], v[i], omega[i], L, R, m, d, m_w
                )
            return f * dt + np.eye(state_size)
        else:
            f = np.zeros((pred_len, state_size, input_size))
            f0 = self.calc_model_jacobian_input(
                L, R, m, d, m_w
            )  # no need to iterate because const.
            for i in range(pred_len):
                f[i, :, :] = f0
            return f * dt