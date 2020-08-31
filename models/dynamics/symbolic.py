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
        self.compute_constants(config.mouse['R'], config.mouse['L'], config.mouse['m'], config.mouse['d'])

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
            s = symbols('s'),
        )

        x, y, R, theta, thetadot, L, m, d, taur, taul, etar, etal, s = self.symbols.values()

        # Define vectors
        state = Matrix([x, y, theta, s])
        tau = Matrix([taur, taul])
        nu = Matrix([etar, etal])
        self.state = state
        self.tau = tau
        self.nu = nu

        # Define Matrices
        Q = Matrix([
            [R/2 * cos(theta), R/2 * cos(theta)],
            [R/2 * sin(theta), R/2 * sin(theta)],
            [R/(2 * L), -R/(2*L)],
            [R/(2 * L), R/(2*L)]
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
        self.xdot_dx = self.xdot.jacobian([x, y, theta, s])
        self.xdot_du = self.xdot.jacobian([taur, taul])

        # vectorize partials
        self.vec_xdot_dx = lambdify(self.symbols.values(), self.xdot_dx)
        self.vec_xdot_du = lambdify(self.symbols.values(), self.xdot_du)


    @staticmethod
    def eval(expression, values):
        ev =  expression.subs([(k,np.float(v)) for k,v in values.items()])
        return np.array(ev).astype(np.float32)

    def compute_constants(self, R, L, m, d):
        """ 
            Computes some constants that are used repetedly in compute funcs below
        """
        self.num1 = L**2 + 2*d**2
        self.denum1 = 2*R**2*d**2*m

        self.r2 = R**2
        self.twoL = 2 * L

    def compute_nudt(self, R, L, m, d, eta_r, eta_l, theta, thetadot, tau_r, tau_l):
        f = np.zeros(2)

        f[0] = (- L**2 + 2*d**2)*(tau_l - (self.r2)*d*eta_r*m*thetadot/(self.twoL))/(self.denum1) + \
                        ( L**2 + 2*d**2)*(tau_r - (self.r2)*d*eta_l*m*thetadot/(self.twoL))/(self.denum1)
        f[1] = (- L**2 + 2*d**2)*(tau_r - (self.r2)*d*eta_l*m*thetadot/(self.twoL))/(self.denum1) + \
                        ( L**2 + 2*d**2)*(tau_l - (self.r2)*d*eta_r*m*thetadot/(self.twoL))/(self.denum1)

        return f

    def compute_xdot(self, R, L, m, d, eta_r, eta_l, theta, thetadot, tau_r, tau_l):
        f = np.zeros(4)

        f[0] = R*((- L**2 + 2*d**2)*(tau_l - (self.r2)*d*eta_r*m*thetadot/(self.twoL))/(self.denum1) + \
                    ( L**2 + 2*d**2)*(tau_r - (self.r2)*d*eta_l*m*thetadot/(self.twoL))/(self.denum1))*np.cos(theta)/2 +\
                    R*((- L**2 + 2*d**2)*(tau_r - (self.r2)*d*eta_l*m*thetadot/(self.twoL))/(self.denum1) + \
                    ( L**2 + 2*d**2)*(tau_l - (self.r2)*d*eta_r*m*thetadot/(self.twoL))/(self.denum1))*np.cos(theta)/2


        f[1] = R*((- L**2 + 2*d**2)*(tau_l - (self.r2)*d*eta_r*m*thetadot/(self.twoL))/(self.denum1) + \
                    ( L**2 + 2*d**2)*(tau_r - (self.r2)*d*eta_l*m*thetadot/(self.twoL))/(self.denum1))*np.sin(theta)/2 + \
                    R*((- L**2 + 2*d**2)*(tau_r - (self.r2)*d*eta_l*m*thetadot/(self.twoL))/(self.denum1) + \
                    ( L**2 + 2*d**2)*(tau_l - (self.r2)*d*eta_r*m*thetadot/(self.twoL))/(self.denum1))*np.sin(theta)/2
                    
        f[2] = R*((- L**2 + 2*d**2)*(tau_l - (self.r2)*d*eta_r*m*thetadot/(self.twoL))/(self.denum1) + \
                    ( L**2 + 2*d**2)*(tau_r - (self.r2)*d*eta_l*m*thetadot/(self.twoL))/(self.denum1))/(self.twoL) - \
                    R*((- L**2 + 2*d**2)*(tau_r - (self.r2)*d*eta_l*m*thetadot/(self.twoL))/(self.denum1) +\
                    ( L**2 + 2*d**2)*(tau_l - (self.r2)*d*eta_r*m*thetadot/(self.twoL))/(self.denum1))/(self.twoL)

        f[3] = R*((- L**2 + 2*d**2)*(tau_l - (self.r2)*d*eta_r*m*thetadot/(self.twoL))/(self.denum1) + \
                    ( L**2 + 2*d**2)*(tau_r - (self.r2)*d*eta_l*m*thetadot/(self.twoL))/(self.denum1))/(self.twoL) + \
                    R*((- L**2 + 2*d**2)*(tau_r - (self.r2)*d*eta_l*m*thetadot/(self.twoL))/(self.denum1) + \
                    ( L**2 + 2*d**2)*(tau_l - (self.r2)*d*eta_r*m*thetadot/(self.twoL))/(self.denum1))/(self.twoL)


        return f


    def compute_xdot_dx(self, shape, R, L, m, d, eta_r, eta_l, theta, thetadot, tau_r, tau_l):
        f = np.zeros(shape)

        f[:, 0, 2] = -R*((- L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(self.denum1) + \
                        ( L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(self.denum1))*np.sin(theta)/2 - \
                        R*((- L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(self.denum1) + \
                        ( L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(self.denum1))*np.sin(theta)/2

        f[:, 1, 2] = R*((- L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(self.denum1) + \
                        ( L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(self.denum1))*np.cos(theta)/2 + \
                        R*((- L**2 + 2*d**2)*(tau_r - R**2*d*eta_l*m*thetadot/(2*L))/(self.denum1) + \
                        ( L**2 + 2*d**2)*(tau_l - R**2*d*eta_r*m*thetadot/(2*L))/(self.denum1))*np.cos(theta)/2

        return f



    def compute_xdot_du(self, shape, R, L, m, d, eta_r, eta_l, theta, thetadot, tau_r, tau_l):
        f = np.zeros(shape)

        f[:, 0, 0] = (-L**2 + 2*d**2)*np.cos(theta)/(4*R*d**2*m) + (L**2 + 2*d**2)*np.cos(theta)/(4*R*d**2*m)

        f[:, 0, 1] = (-L**2 + 2*d**2)*np.cos(theta)/(4*R*d**2*m) + (L**2 + 2*d**2)*np.cos(theta)/(4*R*d**2*m)

        f[:, 1, 0] = (-L**2 + 2*d**2)*np.sin(theta)/(4*R*d**2*m) + (L**2 + 2*d**2)*np.sin(theta)/(4*R*d**2*m)

        f[:, 1, 1] = (-L**2 + 2*d**2)*np.sin(theta)/(4*R*d**2*m) + (L**2 + 2*d**2)*np.sin(theta)/(4*R*d**2*m)

        f[:, 2, 0] = -(-L**2 + 2*d**2)/(4*L*R*d**2*m) + (L**2 + 2*d**2)/(4*L*R*d**2*m)

        f[:, 2, 1] = (-L**2 + 2*d**2)/(4*L*R*d**2*m) - (L**2 + 2*d**2)/(4*L*R*d**2*m)

        f[:, 3, 0] = (-L**2 + 2*d**2)/(4*L*R*d**2*m) + (L**2 + 2*d**2)/(4*L*R*d**2*m)

        f[:, 3, 1] = (-L**2 + 2*d**2)/(4*L*R*d**2*m) + (L**2 + 2*d**2)/(4*L*R*d**2*m)
        return f


