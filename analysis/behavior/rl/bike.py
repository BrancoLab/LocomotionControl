import sys
from matplotlib.pyplot import thetagrids

sys.path.append("./")

from numpy import cos, sin, arctan2, sqrt, pi
from loguru import logger

from utils import inbounds



class Bicycle:
    """
        Implements the dynamical bicicle model.
    """

    # constants
    l_f = 3
    l_r = 2
    width = 1.8
    m_f = 10
    m_r = 12
    c = 4e3

    def __init__(
        self, track, boundaries, u0, delta0, v0, omega0, s, dt
    ):
        self.track = track
        self.boundaries = boundaries
        self.u, self.delta = u0, delta0
        self.v, self.omega = v0, omega0
        self.n, self.psi = 0.0, 0.0
        self.dt = dt
        self.deltadot, self.Fu = 0.0, 0.0
        self.s = s

        # compute constants
        self.m = self.m_f + self.m_r

        # convert units g->Kg, cm->m
        mfKg = self.m_f
        mrKg = self.m_r
        lfM = self.l_f
        lrM = self.l_r

        # compute moment of angular inertia
        self.Iz = mfKg * lfM**2 + mrKg * lrM**2

    def reset(self, u0, delta0, v0, omega0, s):
        """
            Resets the state of the model
        """
        self.u, self.delta = u0, delta0
        self.v, self.omega = v0, omega0
        self.n, self.psi = 0.0, 0.0
        self.s = s

    def k(self):
        """
            Returns track curvature at tje bike's s
        """
        return self.track.curvature(self.s)

    def step(self, deltadot, Fu):
        """
            Simulates the bicycle model for one time step.
        """
        # logger.info(f'Step: deltadot={deltadot:.3f}, Fu={Fu:.3f}')
        delta, u, v, omega, psi, n = (
            self.delta,
            self.u,
            self.v,
            self.omega,
            self.psi,
            self.n,
        )
        c, l_r, l_f = self.c, self.l_r, self.l_f
        m, Iz = self.m, self.Iz

        # compute variables
        beta = arctan2(v , u)  # slip angle
        V = sqrt(u**2 + v**2)

        Ff = c * (delta - (l_f * omega + v) / u)
        Fr = c * (l_r * omega - v) / u

        # compute derivatives
        ds = (V * cos(psi + beta))/(1 - n * self.k())
        dn = u * sin(psi + beta)
        dpsi = omega - self.k() * ds

        ddelta = deltadot
        du = 1 / m * (m * omega * v - Ff * sin(delta) + Fu)
        dv = 1 / m * (-m * omega * u + Ff * cos(delta) + Fr)
        domega = 1 / Iz * (l_f * Ff * cos(delta) - l_r * Fr)


        # update state
        self.s      += ds * self.dt
        self.n      += dn * self.dt
        self.psi    += dpsi * self.dt

        self.delta  += ddelta * self.dt
        self.u      += du * self.dt
        self.v      += dv * self.dt
        self.omega  += domega * self.dt
        
        # enforce boundaries
        self.enforce_boundaries()

        self.deltadot = deltadot
        self.Fu = Fu


    def enforce_boundaries(self):
        """
            Enforces the boundaries on the models variables
        """
        bounds = self.boundaries
        # self.n = inbounds(self.n, bounds["n"].low, bounds["n"].high)
        # self.psi = inbounds(
        #     self.psi, bounds["psi"].low, bounds["psi"].high
        # )
        self.delta = inbounds(
            self.delta, bounds["delta"].low, bounds["delta"].high
        )
        self.u = inbounds(self.u, bounds["u"].low, bounds["u"].high)
        self.v = inbounds(self.v, bounds["v"].low, bounds["v"].high)
        self.omega = inbounds(
            self.omega, bounds["omega"].low, bounds["omega"].high
        )
        return


    @property
    def x(self):
        x, y, theta = self.track.get_at_sval(self.s)
        # print(x, y, theta)
        return x + cos(theta + pi/2) * self.n

    @property
    def y(self):
        x, y, theta = self.track.get_at_sval(self.s)
        return y + sin(theta + pi/2) * self.n

    @property
    def theta(self):
        x, y, theta = self.track.get_at_sval(self.s)
        return theta + self.psi

    def state(self):
        """
            Returns the current state as a dictionary
        """
        return {
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "psi": self.psi,
            "u": self.u,
            "v": self.v,
            "omega": self.omega,
            "delta": self.delta,
            "n": self.n,
            "s": self.s,
        }
