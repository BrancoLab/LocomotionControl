from sympy import *
init_printing()

class Symbolic:
    """
        Collection of functions to do symbolic computations to 
        get the equation of motion of the model + stuff like
        jacobians etc.
        To be subclassed by Model
    """
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

    def get_inverse_dynamics(self):
        """
            If the model is
                x_dot = g + M*tau
            the inverse model is
                tau = M_inv * (x_dot - g)
        """
        # Get variables
        x, y, theta, L, R, m, m_w, d, tau_l, tau_r, v, omega = self.variables.values()
        state = Matrix([x, y, theta, v, omega])

        # Get inverse of M matrix
        M_inv = SparseMatrix(self.matrixes['M']).pinv() # recast as sparse for speed

        # Get inverse model
        self.model_inverse = M_inv * (state - self.matrixes['g'])

        # Vectorize expression
        args = [x, y, theta, v, omega, L, R, m, d, m_w]
        self.calc_inv_dynamics = lambdify(args, self.model_inverse, modules='numpy')



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

