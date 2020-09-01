import numpy as np
from sympy import *
init_printing()
from sympy.solvers import solve
from collections import namedtuple

from proj.model.config import Config
from proj.utils import merge

class Model(Config):
    _M_args = ['theta', 'v', 'omega', 'L', 'R', 'm', 'd', 'm_w' , 'tau_l', 'tau_r']

    _calc_model_jacobian_state_args = ['theta', 'v', 'omega', 'L', 'R', 'm', 'd', 'm_w']
    _calc_model_jacobian_input_args = ['L', 'R', 'm', 'd', 'm_w']

    _control = namedtuple('control', 'tau_r, tau_l')
    _state = namedtuple('state', 'x, y, theta, v, omega')

    def __init__(self):
        Config.__init__(self)

        self._make_simbols()
        self.get_combined_dynamics_kinematics()
        self.get_jacobians()
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

        # Store dxdt model as sympy expression
        self.model = g + M*inp

    def get_jacobians(self):
        x, y, theta, L, R, m, m_w, d, tau_l, tau_r, v, omega = self.variables.values()

        # Get jacobian wrt state
        self.model_jacobian_state = self.model.jacobian([x, y, theta, v, omega])

        # Get jacobian wrt input
        self.model_jacobian_input = self.model.jacobian([tau_r, tau_l])

        # vectorize expressions
        args = [theta, v, omega, L, R, m, d, m_w]
        self.calc_model_jacobian_state = lambdify(args, self.model_jacobian_state, modules='numpy')

        args = [L, R, m, d, m_w]
        self.calc_model_jacobian_input = lambdify(args, self.model_jacobian_input, modules='numpy')


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

    def _fake_step(self, x, u):
        """
            Simulate a step fiven a state and a control
        """
        x = self._state(x)

        # Compute dxdt
        variables = merge(u, x, self.mouse)
        inputs = [variables[a] for a in self._M_args]
        dxdt = self.calc_dqdt(*inputs).ravel()

        # Step
        next_x = np.array(x) + dxdt * self.dt
        return next_x

    def predict_trajectory(self, curr_x, us):
        """
            Compute the trajectory for N steps given a
            state and a (series of) control(s)
        """
        if len(us.shape) == 3:
            raise NotImplementedError

        # get size
        pred_len = us.shape[0]

        # initialze
        x = curr_x # (3,)
        pred_xs = curr_x[np.newaxis, :]

        for t in range(pred_len):
            next_x = self._fake_step(x, us[t])
            # update
            pred_xs = np.concatenate((pred_xs, next_x[np.newaxis, :]), axis=0)
            x = next_x

        return pred_xs

