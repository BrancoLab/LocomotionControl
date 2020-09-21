import numpy as np
from collections import namedtuple

from sympy import (
    Matrix,
    symbols,
    # lambdify,
    cos,
)

from proj.model.model import Model
from proj.model.fast import (
    fast_dqdt_polar,
    fast_model_jacobian_state_polar,
    fast_model_jacobian_input_polar,
)
from rich import print


class ModelPolar(Model):
    MODEL_TYPE = "polar"

    _M_args = [
        "r",
        "gamma",
        "v",
        "omega",
        "L",
        "R",
        "m",
        "d",
        "m_w",
        "tau_l",
        "tau_r",
    ]

    _calc_model_jacobian_state_args = [
        "r",
        "gamma",
        "v",
        "omega",
        "L",
        "R",
        "m",
        "d",
        "m_w",
    ]
    _calc_model_jacobian_input_args = ["L", "R", "m", "d", "m_w"]

    def __init__(self):
        Model.__init__(self, startup=False)

        # add some simbols
        self.variables["r"] = symbols("r", real=True)
        self.variables["gamma"] = symbols("gamma", real=True)

        # Get polar state
        self._state = namedtuple("state", "r, gamma, v, omega")

        # Get model
        self._get_polar_dynamics()
        self.get_jacobians()

    def _get_polar_dynamics(self):
        (
            _,
            _,
            _,
            L,
            R,
            m,
            m_w,
            d,
            tau_l,
            tau_r,
            v,
            omega,
            r,
            gamma,
        ) = self.variables.values()

        # raise ValueError(
        #     "The polar model is wrong, gammadot should depend on v"
        # )
        print(
            "[bold red]\n\n!!! !!! The polar model is wrong, gammadot should depend on v !!! !!! \n"
        )

        # Define moments of inertia
        I_c = m * d ** 2  # mom. inertia around center of gravity
        I_w = m_w * R ** 2  # mom. inertia of wheels
        I = I_c + m * d ** 2 + 2 * m_w * L ** 2 + I_w

        # Define a constant:
        J = I + (2 * I ** 2 / R ** 2) * I_w

        # Define g vector and input vector
        g = Matrix([0, 0, d * omega ** 2, -(m * d * omega * v) / J])
        inp = Matrix([v, omega, tau_r, tau_l])

        # Define M matrix
        M = Matrix(
            [
                [-cos(gamma), 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, L / (m * R), L / (m * R)],
                [0, 0, L / (J * R), -L / (J * R)],
            ]
        )

        # vectorize expression
        # args = [r, gamma, v, omega, L, R, m, d, m_w, tau_l, tau_r]
        # expr = g + M * inp
        # self.calc_dqdt = lambdify(args, expr, modules="numpy")
        self.calc_dqdt = fast_dqdt_polar

        # store matrices
        self.matrixes = dict(g=g, inp=inp, M=M,)

        # Store dxdt model as sympy expression
        self.model = g + M * inp

    def get_jacobians(self):
        (
            _,
            _,
            _,
            L,
            R,
            m,
            m_w,
            d,
            tau_l,
            tau_r,
            v,
            omega,
            r,
            gamma,
        ) = self.variables.values()

        # Get jacobian wrt state
        self.model_jacobian_state = self.model.jacobian([r, gamma, v, omega])

        # Get jacobian wrt input
        self.model_jacobian_input = self.model.jacobian([tau_r, tau_l])

        # vectorize expressions
        # args = [r, gamma, v, omega, L, R, m, d, m_w]
        # self.calc_model_jacobian_state = lambdify(
        #     args, self.model_jacobian_state, modules="numpy"
        # )
        self.calc_model_jacobian_state = fast_model_jacobian_state_polar

        # args = [L, R, m, d, m_w]
        # self.calc_model_jacobian_input = lambdify(
        #     args, self.model_jacobian_input, modules="numpy"
        # )
        self.calc_model_jacobian_input = fast_model_jacobian_input_polar

    def calc_gradient(self, xs, us, wrt="x"):
        """
            Compute the models gradient wrt state or control
        """

        # prep some variables
        r = xs[:, 0]
        gamma = xs[:, 1]
        v = xs[:, 2]
        omega = xs[:, 3]

        L = self.mouse["L"]
        R = self.mouse["R"]
        m = self.mouse["m"]
        m_w = self.mouse["m_w"]
        d = self.mouse["d"]

        # reshapshapee
        (_, state_size) = xs.shape
        (pred_len, input_size) = us.shape

        if wrt == "x":
            f = np.zeros((pred_len, state_size, state_size))
            for i in range(pred_len):
                f[i, :, :] = self.calc_model_jacobian_state(
                    r[i], gamma[i], v[i], omega[i], L, R, m, d, m_w
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
