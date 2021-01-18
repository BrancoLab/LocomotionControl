from sympy import (
    symbols,
    cos,
    sin,
)

import numpy as np
from numba import jit


@jit(nopython=True)
def fast_dxdt(theta, v, omega, L, R, m, d, m_w, tau_l, tau_r, P, N_r, N_l):
    """
        fast implementation of models dyamics
    """
    res = np.zeros(7)

    # xdot
    res[0] = v * np.cos(theta)

    # ydot
    res[1] = v * np.sin(theta)

    # thetadot
    res[2] = omega

    # vdot
    res[3] = (d * m * omega ** 2 + (tau_l + tau_r) / R) / (m + 2 * m_w)

    # omegadot
    res[4] = (L * (-tau_l + tau_r) / R + d * m * omega * v) / (
        4 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m
    )

    # taurdot
    res[5] = -N_r + P

    # tauldot
    res[6] = -N_l + P

    return res


@jit(nopython=True)
def fast_model_jacobian_state(theta, v, omega, L, R, m, d, m_w):
    """
        Fast implementation of the model's derivative wrt to state
        variables. Missing entries are for when the derivative is 0.
    """
    res = np.zeros((7, 7))

    # xdot_wrt_theta
    res[0, 2] = -v * np.sin(theta)

    # xdot_wrt_v
    res[0, 3] = np.cos(theta)

    # ydot_wrt_theta
    res[1, 2] = v * np.cos(theta)

    # ydot_wrt_v
    res[1, 3] = np.sin(theta)

    # vdot_wrt_omega
    res[3, 4] = 2 * d * m * omega / (m + 2 * m_w)

    # vdot_wrt_tau_r
    res[3, 5] = 1 / (R * (m + 2 * m_w))

    # vdot_wrt_tau_l
    res[3, 6] = 1 / (R * (m + 2 * m_w))

    # omegadot_wrt_v
    res[4, 3] = (
        d * m * omega / (4 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m)
    )

    # omegadot_wrt_omega
    res[4, 4] = d * m * v / (4 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m)

    # omegadot_wrt_tau_r
    res[4, 5] = L / (R * (4 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m))

    # omegadot_wrt_tau_l
    res[4, 6] = -L / (R * (4 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m))

    return res


@jit(nopython=True)
def fast_model_jacobian_input(L, R, m, d, m_w):
    """
        Fast implementation of the model's derivative wrt to
        the inputs (controls).
        Missing entries is for when the derivative is 0
    """
    res = np.zeros((7, 3))

    # taurdot_wrt_P
    res[5, 0] = 1
    # taurdot_wrt_N_r
    res[5, 2] = -1
    # tauldot_wrt_P
    res[6, 0] = 1
    # tauldot_wrt_N_l
    res[6, 1] = -1

    return res


class ModelDynamics(object):
    # Names of arguments of the dynamic's M matrix
    _M_args = [
        "theta",
        "v",
        "omega",
        "L",
        "R",
        "m",
        "d",
        "m_w",
        "tau_l",
        "tau_r",
        "P",
        "N_r",
        "N_l",
    ]

    def __init__(self,):
        """
            This model uses sympy to create expression to compute
            the cartesian model's dynamics, inverse dynamics and 
            wheel dynamics. 
            These expressions are then used to compute the model's movements
            as controls are applied to it.
        """
        self._make_simbols()

        # Get numba expressions for the dynamics
        self.calc_dxdt = fast_dxdt
        self.calc_model_jacobian_state = fast_model_jacobian_state
        self.calc_model_jacobian_input = fast_model_jacobian_input

        # to get sympy expressions to compute dynamics:
        # self.get_combined_dynamics_kinematics()
        # self.get_jacobians()

    def _make_simbols(self):
        """
            Create sympy symbols
        """
        # state variables
        x, y, theta, v, omega = symbols("x, y, theta, v, omega", real=True)
        tau_r, tau_l = symbols("tau_r, tau_l", real=True)

        # static variables
        L, R, m, m_w, d = symbols("L, R, m, m_w, d", real=True)

        # control variables
        P, N_r, N_l = symbols("P, N_r, N_l", real=True)

        # store symbols
        self.variables = dict(
            x=x,
            y=y,
            theta=theta,
            L=L,
            R=R,
            m=m,
            m_w=m_w,
            d=d,
            tau_l=tau_l,
            tau_r=tau_r,
            v=v,
            omega=omega,
            P=P,
            N_l=N_l,
            N_r=N_r,
        )

    def get_combined_dynamics_kinematics(self):
        """
            Sets up the matrix representation of a
            system of differential equations for
            how each variable varies over time.
        """
        (
            x,
            y,
            theta,
            L,
            R,
            m,
            m_w,
            d,
            tau_l,
            tau_r,
            v,
            omega,
            P,
            N_l,
            N_r,
        ) = self.variables.values()

        # Define moments of inertia
        I_c = m * d ** 2  # mom. inertia around center of gravity
        I_w = m_w * R ** 2  # mom. inertia of wheels
        I = I_c + m * d ** 2 + 2 * m_w * L ** 2 + I_w

        # define differential equations
        self.equations = dict(
            xdot=v * cos(theta),
            ydot=v * sin(theta),
            thetadot=omega,
            vdot=(
                ((1 / R * (tau_r + tau_l)) + (m * d * omega ** 2))
                / (m + (2 * I_w) / R ** 2)
            ),
            omegadot=(
                ((L / R * (tau_r - tau_l)) + (m * d * omega * v))
                / (I + (((2 * L ** 2) / R ** 2) * I_w))
            ),
            taurdot=P - N_r,
            tauldot=P - N_l,
        )

    def get_jacobians(self):
        (
            x,
            y,
            theta,
            L,
            R,
            m,
            m_w,
            d,
            tau_l,
            tau_r,
            v,
            omega,
            P,
            N_l,
            N_r,
        ) = self.variables.values()

        # Get jacobian wrt state
        vrs = [x, y, theta, v, omega, tau_r, tau_l]
        self.model_jacobian_state = {}
        for eqname, eq in self.equations.items():
            for wrt in vrs:
                name = f"{eqname}_wrt_{wrt}"
                self.model_jacobian_state[name] = eq.diff(wrt)

        # Get jacobian wrt input
        vrs = [P, N_l, N_r]
        self.model_jacobian_input = {}
        for eqname, eq in self.equations.items():
            for wrt in vrs:
                name = f"{eqname}_wrt_{wrt}"
                self.model_jacobian_input[name] = eq.diff(wrt)
