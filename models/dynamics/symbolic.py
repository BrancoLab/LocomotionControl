import numpy as np
from sympy import symbols, Matrix, diff, sin, cos, lambdify, init_printing, latex
init_printing() 


class Symbolic():
    """
        Handles symbolic operations to facilitate derivation
    """
    def __init__(self, config):
        self.config = config
        mouse = config.mouse

        # Define symbols
        self.symbols = dict(
            x = symbols('x'),
            y = symbols('y'),
            R = symbols('R'),
            theta = symbols('theta'),
            thetadot = symbols('thetadot'),
            L = symbols('L'),
            m = symbols('m'),
            d = symbols('d'),
            taur = symbols('tau_r'),
            taul = symbols('tau_l'),
            etar = symbols('eta_r'),
            etal = symbols('eta_l'),
        )

        x, y, R, theta, thetadot, L, m, d, taur, taul, etar, etal = self.symbols.values()

        # Define vectors
        state = Matrix([x, y, theta])
        tau = Matrix([taur, taul])
        nu = Matrix([etar, etal])
        self.state = state
        self.tau = tau
        self.nu = nu

        # Define Matrices
        Q = Matrix([
            [R/2 * cos(theta), R/2 * cos(theta)],
            [R/2 * sin(theta), R/2 * sin(theta)],
            [R/(2 * L), -R/(2*L)]
        ])
        self.Q = Q

        M = Matrix([
            [(R**2/(4*L**2))*(m*L**2 + 2*m*d**2), (R**2/(4*L**2))*(m*L**2 - 2*m*d**2)],
            [(R**2/(4*L**2))*(m*L**2 - 2*m*d**2), (R**2/(4*L**2))*(m*L**2 + 2*m*d**2)]
        ])
        self.M = M

        B = Matrix([
            [1, 0],
            [0, 1]
        ])
        self.B = B


        V = Matrix([
            [0, R**2/(2*L)*m*d*thetadot],
            [R**2/(2*L)*m*d*thetadot, 0],
        ])
        self.V = V

        # Compute dxdt and dnudt
        self.dnudt = M**-1 * (B * tau - V * nu)
        self.xdot = Q * self.dnudt        

        # Compute partial derivs
        self.xdot_dx = Matrix(np.zeros((3, 3)))

        for row in range(3):
            for col in range(3):
                self.xdot_dx[row, col] = diff(self.xdot[row], state[col])


        self.xdot_du = Matrix(np.zeros((3, 2)))
        for row in range(3):
            for col in range(2):
                self.xdot_du[row, col] = diff(self.xdot[row], tau[col])

        # vectorize partials
        self.vec_xdot_dx = lambdify(self.symbols.values(), self.xdot_dx)
        self.vec_xdot_du = lambdify(self.symbols.values(), self.xdot_du)


    @staticmethod
    def eval(expression, values):
        ev =  expression.subs([(k,np.float(v)) for k,v in values.items()])
        return np.array(ev).astype(np.float32)





    def compute_xdot(self, R, L, m, d, eta_r, eta_l, theta, thetadot, tau_r, tau_l):
        f = np.zeros(3)


        f[0] = R*((-L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(2*R**2*d**2*m) + \
                    (L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(2*R**2*d**2*m))*np.cos(theta)/2 + \
                    R*((-L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(2*R**2*d**2*m) + \
                    (L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(2*R**2*d**2*m))*np.cos(theta)/2

        f[1] = R*((-L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(2*R**2*d**2*m) + \
                    (L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(2*R**2*d**2*m))*np.sin(theta)/2 + \
                    R*((-L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(2*R**2*d**2*m) + \
                    (L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(2*R**2*d**2*m))*np.sin(theta)/2

        f[2] = R*((-L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(2*R**2*d**2*m) + \
                    (L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(2*R**2*d**2*m))/(2*L) - \
                    R*((-L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(2*R**2*d**2*m) + \
                    (L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(2*R**2*d**2*m))/(2*L)

        return f


    def compute_xdot_dx(self, shape, R, L, m, d, eta_r, eta_l, theta, thetadot, tau_r, tau_l):
        f = np.zeros(shape)

        f[:, 0, 2] = -R*((-L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(2*R**2*d**2*m) +\
                        (L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(2*R**2*d**2*m))*np.sin(theta)/2 - \
                        R*((-L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(2*R**2*d**2*m) +\
                        (L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(2*R**2*d**2*m))*np.sin(theta)/2

        f[:, 1, 2] = R*((-L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(2*R**2*d**2*m) +\
                        (L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(2*R**2*d**2*m))*np.cos(theta)/2 + \
                        R*((-L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(2*R**2*d**2*m) +\
                        (L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(2*R**2*d**2*m))*np.cos(theta)/2
        return f



    def compute_xdot_du(self, shape, R, L, m, d, eta_r, eta_l, theta, thetadot, tau_r, tau_l):
        f = np.zeros(shape)

        f[:, 0, 0] = (-L**2 + 2*d**2)*np.cos(theta)/(4*R*d**2*m) + (L**2 + 2*d**2)*np.cos(theta)/(4*R*d**2*m)
        f[:, 0, 1] = (-L**2 + 2*d**2)*np.cos(theta)/(4*R*d**2*m) + (L**2 + 2*d**2)*np.cos(theta)/(4*R*d**2*m)
        f[:, 1, 0] = (-L**2 + 2*d**2)*np.sin(theta)/(4*R*d**2*m) + (L**2 + 2*d**2)*np.sin(theta)/(4*R*d**2*m)
        f[:, 1, 1] = (-L**2 + 2*d**2)*np.sin(theta)/(4*R*d**2*m) + (L**2 + 2*d**2)*np.sin(theta)/(4*R*d**2*m)
        f[:, 2, 0] = -(-L**2 + 2*d**2)/(4*L*R*d**2*m) + (L**2 + 2*d**2)/(4*L*R*d**2*m)
        f[:, 2, 1] = (-L**2 + 2*d**2)/(4*L*R*d**2*m) - (L**2 + 2*d**2)/(4*L*R*d**2*m)

        return f


