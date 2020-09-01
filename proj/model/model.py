import numpy as np
from sympy import *
init_printing()
from sympy.solvers import solve
from collections import namedtuple

from proj.model.config import Config
from proj.utils import merge

class Model(Config):
    _D_args = ['R', 'd', 'm', 'm_w', 'omega', 'tau_r', 'tau_l', "L", 'v']

    _control = namedtuple('control', 'tau_r, tau_l')
    _eta = namedtuple('eta', 'v, omega')
    _etadot = namedtuple('etadot', 'vdot, omegadot')
    _state = namedtuple('state', 'x, y, theta, s')

    def __init__(self):
        Config.__init__(self)

        self._make_simbols()
        self.get_dynamics()
        self.get_kinematics()
        self.reset()

    def reset(self):
        self.curr_x = self._state(0, 0, 0, 0)
        self.curr_thetadot = {'thetadot': 0}
        self.curr_eta = self._eta(0, 0)
        self.curr_etadot = self._etadot(0, 0)
        self.curr_control = self._control(0, 0) # use only to keep track

        self.history = dict(
            x = [],
            y = [],
            theta = [],
            s = [],
            v = [],
            omega = [],
            tau_r = [],
            tau_l = [],
        )

        self._append_history() # make sure first state is included

    def _append_history(self):
        for ntuple in [self.curr_x, self.curr_eta, self.curr_control]:
            for k,v in ntuple._asdict().items():
                self.history[k].append(v)

    def _make_simbols(self):
        # state variables
        x, y, theta, thetadot, s = symbols('x, y, theta, thetadot, s', real=True)

        # static variables
        L, R, m, m_w, d = symbols('L, R, m, m_w, d', real=True)

        # wheel speeds
        psi_l, psi_r = symbols('psi_l, psi_r', real=True)
        psidot_l, psidot_r = symbols('psidot_l, psidot_r', real=True)

        # control variables
        tau_r, tau_l = symbols('tau_r, tau_l', real=True)

        # speeds
        v, omega = symbols('v, omega', real=True)
        vdot, omegadot = symbols('vdot, omegadot', real=True)

        self.variables = dict(
            x = x,
            y = y,
            theta = theta,
            thetadot = thetadot,
            s = s,

            L = L,
            R = R,
            m = m,
            m_w = m_w,
            d = d,

            psi_l=psi_l,
            psi_r = psi_r,
            psidot_l=psidot_l,
            psidot_r = psidot_r,

            tau_l=tau_l,
            tau_r=tau_r,

            v=v,
            omega=omega,
            vdot=vdot,
            omegadot=omegadot,
        )

    # def _make_matrices(self): 
    #     x, y, theta, thetadot, s, L, R, m, d, psi_l, psi_r, psidot_l, \
    #                 psidot_r, tau_l, tau_r, v, omega, vdot, omegadot = self.variables.values()
        
    #     # Define moments of inertia
    #     I = 2 * m * d**2

    #     # Define 
    #     F = R**2/(4*L**2)

    #     Q = Matrix([
    #         [R/2 * cos(theta), R/2 * cos(theta)],
    #         [R/2 * sin(theta), R/2 * sin(theta)],
    #         [R/(2 * L), -R/(2*L)],
    #         [R/(2 * L), R/(2*L)]
    #     ])

    #     M = Matrix([
    #         [F*(m*L**2 + I), F*(m*L**2 - I)],
    #         [F*(m*L**2 - I), F*(m*L**2 + I)]
    #     ])

    #     V = Matrix([
    #         [0, R**2/(2*L)*m*d*thetadot],
    #         [-R**2/(2*L)*m*d*thetadot, 0],
    #     ])

    #     tau = Matrix([tau_r, tau_l])
    #     eta = Matrix([psi_r, psi_l])
    #     etadot = Matrix([psidot_r, psidot_l])

    #     self.matrixes = dict(
    #         Q = Q,
    #         M = M,
    #         V = V,
    #         eta = eta,
    #         etadot = etadot,
    #         tau = tau,
    #     )

    #     return Q, M, V, tau, eta, etadot

    def get_dynamics(self):
        x, y, theta, thetadot, s, L, R, m, m_w, d, psi_l, psi_r, psidot_l, \
                psidot_r, tau_l, tau_r, v, omega, vdot, omegadot = self.variables.values()
        self.matrixes = {}

        # Define moments of inertia
        I_c = m * d**2 # mom. inertia around center of gravity
        I_w = m_w * R**2 # mom. inertia of wheels
        I = I_c + m*d**2 + 2*m_w*L**2 + I_w

        # Define two diff equations
        eq1 = Eq(m * vdot - m*d*omega**2, 1/R * (tau_r + tau_l))
        eq2 = Eq(I*omegadot + m*d*omega*v, (L/R)*(tau_r - tau_l))

        # solve
        vdot_sol = solve(eq1, vdot)[0]
        omegadot_sol = solve(eq2, omegadot)[0]

        # store in matrix
        D = Matrix([vdot_sol, omegadot_sol])
        self.matrixes['D'] = D

        # Create lambdified function
        self.calc_D = lambdify([self.variables[k] for k in self._D_args], self.matrixes['D'])

    def get_kinematics(self):
        x, y, theta, thetadot, s, L, R, m, m_w, d, psi_l, psi_r, psidot_l, \
                psidot_r, tau_l, tau_r, v, omega, vdot, omegadot = self.variables.values()

        Q = Matrix([
            [cos(theta), 0],
            [sin(theta), 0],
            [0, 1],
            [1, 0]
        ])
        self.matrixes['Q'] = Q

        x = Q * Matrix([v, omega])

        self.matrixes['Qeta'] = x

        self.calc_Q = lambdify([theta, v, omega], x)


    def step(self, u):
        # First update eta
        variables = merge(u, self.curr_x, self.curr_thetadot, self.mouse, self.curr_eta, self.curr_etadot)

        # dv, domega
        d = self.calc_D(*[variables[a] for a in self._D_args]).ravel()
        self.curr_etadot = self._etadot(*np.array(d))

        self.curr_eta = self._eta(*np.array(self.curr_eta) + d * self.dt)

        # Then use deta to compute the state
        dxdt = self.calc_Q(variables['theta'], self.curr_eta.v, self.curr_eta.omega).ravel()
        self.curr_x = self._state(*(np.array(self.curr_x) + dxdt * self.dt))

        # Update history
        self._append_history()

