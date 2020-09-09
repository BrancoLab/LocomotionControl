import numpy as np
from numba import jit

"""
    Faster computation with numba
"""


# ---------------------------------------------------------------------------- #
#                                  POLAR MODEL                                 #
# ---------------------------------------------------------------------------- #
@jit(nopython=True)
def fast_dqdt_polar(r, gamma, v, omega, L, R, m, d, m_w, tau_l, tau_r):
    _r = -v * np.cos(gamma)
    _gamma = -omega
    _v = L * tau_l / (R * m) + L * tau_r / (R * m) + d * omega ** 2
    _omega = (
        -L
        * tau_l
        / (
            R
            * (
                2 * L ** 2 * m_w
                + R ** 2 * m_w
                + 2 * d ** 2 * m
                + 2
                * m_w
                * (2 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m) ** 2
            )
        )
        + L
        * tau_r
        / (
            R
            * (
                2 * L ** 2 * m_w
                + R ** 2 * m_w
                + 2 * d ** 2 * m
                + 2
                * m_w
                * (2 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m) ** 2
            )
        )
        - d
        * m
        * omega
        * v
        / (
            2 * L ** 2 * m_w
            + R ** 2 * m_w
            + 2 * d ** 2 * m
            + 2 * m_w * (2 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m) ** 2
        )
    )

    return np.array([_r, _gamma, _v, _omega])


@jit(nopython=True)
def fast_model_jacobian_state_polar(r, gamma, v, omega, L, R, m, d, m_w):
    res = np.zeros((4, 4))

    res[0, 1] = v * np.sin(gamma)
    res[0, 2] = -np.cos(gamma)
    res[1, 3] = -1
    res[2, 3] = 2 * d * omega
    res[3, 2] = (
        -d
        * m
        * omega
        / (
            2 * L ** 2 * m_w
            + R ** 2 * m_w
            + 2 * d ** 2 * m
            + 2 * m_w * (2 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m) ** 2
        )
    )
    res[3, 3] = (
        -d
        * m
        * v
        / (
            2 * L ** 2 * m_w
            + R ** 2 * m_w
            + 2 * d ** 2 * m
            + 2 * m_w * (2 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m) ** 2
        )
    )

    return res


@jit(nopython=True)
def fast_model_jacobian_input_polar(L, R, m, d, m_w):
    res = np.zeros((4, 2))

    res[2, 0] = L / (R * m)
    res[2, 1] = L / (R * m)
    res[3, 0] = L / (
        R
        * (
            2 * L ** 2 * m_w
            + R ** 2 * m_w
            + 2 * d ** 2 * m
            + 2 * m_w * (2 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m) ** 2
        )
    )
    res[3, 1] = -L / (
        R
        * (
            2 * L ** 2 * m_w
            + R ** 2 * m_w
            + 2 * d ** 2 * m
            + 2 * m_w * (2 * L ** 2 * m_w + R ** 2 * m_w + 2 * d ** 2 * m) ** 2
        )
    )

    return res
