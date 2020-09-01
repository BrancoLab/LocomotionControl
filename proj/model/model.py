import numpy as np
from sympy import *
init_printing()
from sympy.solvers import solve
from collections import namedtuple

from proj.model.config import Config
from proj.utils import merge

class Model(Config):
    _M_args = ['theta', 'v', 'omega', 'L', 'R', 'm', 'd', 'm_w' , 'tau_l', 'tau_r']

    _control = namedtuple('control', 'tau_r, tau_l')
    _state = namedtuple('state', 'x, y, theta, v, omega')

    def __init__(self):
        Config.__init__(self)

        self._make_simbols()
        self.get_combined_dynamics_kinematics()
        self.reset()

    def reset(self):
        self.curr_x = self._state(0, 0, 0, 0, 0)
        self.curr_control = self._control(0, 0) # use only to keep track

        self.history = dict(
            x = [],
            y = [],
            theta = [],
            v = [],
            omega = [],
            tau_r = [],
            tau_l = [],
        )

        self._append_history() # make sure first state is included

    def _append_history(self):
        for ntuple in [self.curr_x, self.curr_control]:
            for k,v in ntuple._asdict().items():
                self.history[k].append(v)


    def _make_simbols(self):
        # state variables
        x, y, theta, thetadot = symbols('x, y, theta, thetadot', real=True)

        # static variables
        L, R, m, m_w, d = symbols('L, R, m, m_w, d', real=True)

        # control variables
        tau_r, tau_l = symbols('tau_r, tau_l', real=True)

        # speeds
        v, omega = symbols('v, omega', real=True)
        vdot, omegadot = symbols('vdot, omegadot', real=True)

        # store symbols
        self.variables = dict(
            x = x,
            y = y,
            theta = theta,

            L = L,
            R = R,
            m = m,
            m_w = m_w,
            d = d,

            tau_l=tau_l,
            tau_r=tau_r,

            v=v,
            omega=omega,
        )



    def get_combined_dynamics_kinematics(self):
        x, y, theta, L, R, m, m_w, d, tau_l, tau_r, v, omega = self.variables.values()

        # Define moments of inertia
        I_c = m * d**2 # mom. inertia around center of gravity
        I_w = m_w * R**2 # mom. inertia of wheels
        I = I_c + m*d**2 + 2*m_w*L**2 + I_w

        # Define a constant:
        J = I + (2*I**2/R**2) * I_w

        # Define g vector and input vector
        g = Matrix([0, 0, 0, d*omega**2, - (m * d * omega * v)/J])
        inp = Matrix([v, omega, tau_r, tau_l])

        # Define M matrix
        M = Matrix([
            [cos(theta), 0, 0, 0],
            [sin(theta), 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, L/(m*R), L/(m*R)],
            [0, 0, L/(J*R), -L/(J*R)]
        ])

        # vectorize expression
        args = [theta, v, omega, L, R, m, d, m_w , tau_l, tau_r]
        expr = g + M*inp
        self.calc_dqdt = lambdify(args, expr, modules='numpy')

        # store matrices
        self.matrixes = dict(
            g = g,
            inp = inp,
            M = M,
        )


    def step(self, u):
        # Compute dxdt
        variables = merge(u, self.curr_x, self.mouse)
        inputs = [variables[a] for a in self._M_args]
        dxdt = self.calc_dqdt(*inputs).ravel()

        # Step
        next_x = np.array(self.curr_x) + dxdt * self.dt
        self.curr_x = self._state(*next_x)

        # Update history
        self._append_history()

