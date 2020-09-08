import numpy as np
from collections import namedtuple

from sympy import (
    Matrix,
    symbols,
    lambdify,
    cos,
    sqrt,
    acos,
    sign,
)

from proj.model.model import Model


class ModelPolar(Model):
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

        # Kinematics
        r1 = sqrt(r ** 2 + v ** 2 - 2 * r * v * cos(gamma))
        rdot = r1 - r

        num = r ** 2 + r1 ** 2 - v ** 2
        den = 2 * r * r1
        gammadot = (acos(num / den) + omega * (-sign(gamma))).simplify()

        # lambdify kinematics
        self.rdot = rdot
        self.calc_rdot = lambdify((r, v, gamma), rdot, modules="numpy")

        self.gammadot = gammadot
        self.calc_gammadot = lambdify(
            (r, v, gamma, omega), gammadot, modules="numpy"
        )

        # Get reduced dynamics
        vdot = (
            self.matrixes["g"][3]
            * self.matrixes["M"][3, 2:]
            * Matrix(self.matrixes["inp"][2:])
        )
        omegadot = (
            self.matrixes["g"][4]
            * self.matrixes["M"][4, 2:]
            * Matrix(self.matrixes["inp"][2:])
        )

        # lambdify dynamics
        self.vdot = vdot
        self.calc_vdot = lambdify(
            (L, d, m, R, omega, tau_l, tau_r), vdot, modules="numpy"
        )

        self.omegadot = omegadot
        self.calc_omegadot = lambdify(
            (L, d, m, R, m_w, omega, tau_l, tau_r), omegadot, modules="numpy"
        )

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

        # ---------------------------- Jacobian wrt state ---------------------------- #
        self.model_jacobian_state = Matrix(np.zeros((4, 4)))

        # wrt gammadot
        self.model_jacobian_state[0, 0] = self.gammadot.diff(gamma)
        self.model_jacobian_state[0, 1] = self.gammadot.diff(r)
        self.model_jacobian_state[0, 2] = self.gammadot.diff(v)
        self.model_jacobian_state[0, 3] = self.gammadot.diff(omega)

        # wrt rdot
        self.model_jacobian_state[1, 0] = self.rdot.diff(gamma)
        self.model_jacobian_state[1, 1] = self.rdot.diff(r)
        self.model_jacobian_state[1, 2] = self.rdot.diff(v)
        self.model_jacobian_state[1, 3] = self.rdot.diff(omega)

        # wrt vdot
        self.model_jacobian_state[2, 0] = self.vdot.diff(gamma)
        self.model_jacobian_state[2, 1] = self.vdot.diff(r)
        self.model_jacobian_state[2, 2] = self.vdot.diff(v)
        self.model_jacobian_state[2, 3] = self.vdot.diff(omega)

        # wrt omegadot
        self.model_jacobian_state[3, 0] = self.omegadot.diff(gamma)
        self.model_jacobian_state[3, 1] = self.omegadot.diff(r)
        self.model_jacobian_state[3, 2] = self.omegadot.diff(v)
        self.model_jacobian_state[3, 3] = self.omegadot.diff(omega)

        # ---------------------------- Jacobian wrt input ---------------------------- #
        self.model_jacobian_input = Matrix(np.zeros((4, 2)))

        # wrt gammadot
        self.model_jacobian_input[0, 0] = self.gammadot.diff(tau_r)
        self.model_jacobian_input[0, 1] = self.gammadot.diff(tau_l)

        # wrt rdot
        self.model_jacobian_input[1, 0] = self.rdot.diff(tau_r)
        self.model_jacobian_input[1, 1] = self.rdot.diff(tau_l)

        # wrt vdot
        self.model_jacobian_input[2, 0] = self.vdot.diff(tau_r)
        self.model_jacobian_input[2, 1] = self.vdot.diff(tau_l)

        # wrt omegadot
        self.model_jacobian_input[3, 0] = self.omegadot.diff(tau_r)
        self.model_jacobian_input[3, 1] = self.omegadot.diff(tau_l)

        # --------------------------------- vectorize --------------------------------- #
        args = [r, gamma, v, omega, L, R, m, d, m_w]
        self.calc_model_jacobian_state = lambdify(
            args, self.model_jacobian_state, modules="numpy"
        )

        args = [r, gamma, v, omega, L, R, m, d, m_w]
        self.calc_model_jacobian_input = lambdify(
            args, self.model_jacobian_input, modules="numpy"
        )

    def calc_dqdt(self):

        # calc dynamics and kinematics

        return
