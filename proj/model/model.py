import numpy as np
from collections import namedtuple

from proj.model.config import Config
from proj.utils.misc import merge
from proj.model.dynamics import ModelDynamics


class Model(Config, ModelDynamics):
    _control = namedtuple("control", "tau_r, tau_l")
    _state = namedtuple("state", "x, y, theta, v, omega")
    _goal = namedtuple(
        "state", "goal_x, goal_y, goal_theta, goal_v, goal_omega"
    )
    _dxdt = namedtuple("dxdt", "x_dot, y_dot, theta_dot, v_dot, omega_dot")
    _wheel_state = namedtuple("wheel_state", "nudot_right, nudot_left")

    def __init__(self, trial_n=0):
        Config.__init__(self)
        ModelDynamics.__init__(self)

        self.trajectory["trial_n"] = trial_n

    def reset(self):
        self.history = dict(
            x=[],
            y=[],
            theta=[],
            v=[],
            omega=[],
            goal_x=[],
            goal_y=[],
            goal_theta=[],
            goal_v=[],
            goal_omega=[],
            tau_r=[],
            tau_l=[],
            r=[],
            gamma=[],
            trajectory_idx=[],
            nudot_left=[],  # acceleration of left wheel
            nudot_right=[],  # acceleration of right wheel
        )

    def _append_history(self):
        for ntuple in [
            self.curr_x,
            self.curr_control,
            self.curr_wheel_state,
            self.curr_goal,
        ]:
            for k, v in ntuple._asdict().items():
                self.history[k].append(v)

        self.history["trajectory_idx"].append(
            self.curr_traj_waypoint_idx
        )  # this is updated by env.plan

    def step(self, u, curr_goal):
        # prep some variables
        self.curr_x = self._state(*self.curr_x)
        self.curr_goal = self._goal(*curr_goal)
        u = self._control(*np.array(u))

        variables = merge(u, self.curr_x, self.mouse)
        inputs = [variables[a] for a in self._M_args]

        # Compute wheel accelerations
        w = self.calc_wheels_accels(
            variables["L"],
            variables["R"],
            variables["theta"],
            self.curr_x.v,
            self.curr_x.omega,
        )
        self.curr_wheel_state = self._wheel_state(*w.ravel())

        # Update history
        self.curr_control = u
        self._append_history()

        # Compute dxdt
        dxdt = self.calc_dqdt(*inputs).ravel()
        self.curr_dxdt = self._dxdt(*dxdt)

        if np.any(np.isnan(dxdt)) or np.any(np.isinf(dxdt)):
            raise ValueError("Nans in dxdt")

        # Step
        next_x = np.array(self.curr_x) + dxdt * self.dt
        self.curr_x = self._state(*next_x)

    # ------------------------------ Control related ----------------------------- #

    def _fake_step(self, x, u):
        """
            Simulate a step fiven a state and a control
        """
        x = self._state(*x)
        u = self._control(*u)

        # Compute dxdt
        variables = merge(u, x, self.mouse)
        inputs = [variables[a] for a in self._M_args]
        dxdt = self.calc_dqdt(*inputs).ravel()

        if np.any(np.isnan(dxdt)) or np.any(np.isinf(dxdt)):
            # raise ValueError('Nans in dxdt')
            print("nans in dxdt during fake step")

        # Step
        next_x = np.array(x) + dxdt * self.dt
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

        L = self.mouse["L"]
        R = self.mouse["R"]
        m = self.mouse["m"]
        m_w = self.mouse["m_w"]
        d = self.mouse["d"]

        (_, state_size) = xs.shape
        (pred_len, input_size) = us.shape

        if wrt == "x":
            f = np.zeros((pred_len, state_size, state_size))
            for i in range(pred_len):
                f[i, :, :] = self.calc_model_jacobian_state(
                    theta[i], v[i], omega[i], L, R, m, d, m_w
                )
            return f * self.dt + np.eye(state_size)
        else:
            f = np.zeros((pred_len, state_size, input_size))
            f0 = self.calc_model_jacobian_input(
                L, R, m, d, m_w
            )  # no need to iterate because const.
            for i in range(pred_len):
                f[i, :, :] = f0
            return f * self.dt

    # def calc_angle_distance_from_goal(self, goal_x, goal_y):
    #     # Get current position
    #     x, y = self.curr_x.x, self.curr_x.y

    #     gamma = np.arctan2(goal_y - y, goal_x - x)
    #     gamma = fit_angle_in_range(gamma, is_deg=False)
    #     gamma -= self.curr_x.theta

    #     # compute distance
    #     r = calc_distance_between_points_2d([x, y], [goal_x, goal_y])

    #     return r, gamma
