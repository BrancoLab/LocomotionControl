import sys

sys.path.append("./")

from numpy import cos, sin, arctan2, sqrt
from loguru import logger

from utils import inbounds



class Bicycle:
    """
        Implements the dynamical bicicle model.
    """

    # constants
    l_f = 4
    l_r = 3
    width = 2.0
    m_f = 12
    m_r = 12
    c = 1e4

    def __init__(
        self, track, boundaries, x0, y0, u0, delta0, v0, theta0, omega0, dt
    ):
        self.track = track
        self.boundaries = boundaries
        self.x, self.y = x0, y0
        self.u, self.delta = u0, delta0
        self.v, self.theta, self.omega = v0, theta0, omega0
        self.n, self.psi = 0.0, 0.0
        self.dt = dt
        self.deltadot, self.Fu = 0.0, 0.0

        # compute constants
        self.m = self.m_f + self.m_r

        # convert units g->Kg, cm->m
        mfKg = self.m_f # / 100
        mrKg = self.m_r # / 100
        lfM = self.l_f # / 100
        lrM = self.l_r # / 100

        # compute moment of angular inertia
        self.Iz = mfKg * lfM**2 + mrKg * lrM**2

    def reset(self, x0, y0, u0, delta0, v0, theta0, omega0):
        """
            Resets the state of the model
        """
        self.x, self.y = x0, y0
        self.u, self.delta = u0, delta0
        self.v, self.theta, self.omega = v0, theta0, omega0
        self.n, self.psi = 0.0, 0.0

    def s(self):
        """
            Returns current track progress
        """
        return self.track.s(self.x, self.y)

    def k(self):
        """
            Returns curvature
        """
        return self.track.curvature(self.s())

    def step(self, deltadot, Fu):
        """
            Simulates the bicycle model for one time step.
        """

        theta, delta, u, v, omega = (
            self.theta,
            self.delta,
            self.u,
            self.v,
            self.omega,
        )
        psi = self.psi
        c, l_r, l_f = self.c, self.l_r, self.l_f
        m, Iz = self.m, self.Iz

        # compute variables
        beta = arctan2(v , u)  # slip angle
        V = sqrt(u**2 + v**2)

        Ff = c * (delta - (l_f * omega + v) / u)
        Fr = c * (l_r * omega - v) / u

        # compute derivatives
        dn = u * sin(psi + beta)
        dpsi = omega - self.k()

        ddelta = deltadot
        du = 1 / m * (m * omega * v - Ff * sin(delta) + Fu)
        dv = 1 / m * (-m * omega * u + Ff * cos(delta) + Fr)
        domega = 1 / Iz * (l_f * Ff * cos(delta) - l_r * Fr)

        # logger.info(f'du: {du:.4f}, ddelta: {ddelta:.4f}, domega: {domega:.4f}')

        # update state
        self.x += V * cos(beta + theta) * self.dt
        self.y += V * sin(beta + theta) * self.dt
        self.theta += omega * self.dt

        self.n += dn * self.dt
        self.psi += dpsi * self.dt

        self.delta += ddelta * self.dt
        self.u += du * self.dt
        self.v += dv * self.dt
        self.omega += domega * self.dt
        
        # enforce boundaries
        # self.enforce_boundaries()

        self.deltadot = deltadot
        self.Fu = Fu


    def enforce_boundaries(self):
        """
            Enforces the boundaries on the models variables
        """
        bounds = self.boundaries
        self.n = inbounds(self.n, bounds["n"].low, bounds["n"].high)
        self.psi = inbounds(
            self.psi, bounds["psi"].low, bounds["psi"].high
        )
        self.delta = inbounds(
            self.delta, bounds["delta"].low, bounds["delta"].high
        )
        self.u = inbounds(self.u, bounds["u"].low, bounds["u"].high)
        self.v = inbounds(self.v, bounds["v"].low, bounds["v"].high)
        self.omega = inbounds(
            self.omega, bounds["omega"].low, bounds["omega"].high
        )
        return


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
        }
