from sympy import (
    Matrix,
    symbols,
    init_printing,
    lambdify,
    cos,
    sin,
    SparseMatrix,
)

from proj.model._dynamics import (
    fast_dqdt,
    fast_model_jacobian_state,
    fast_model_jacobian_input,
)


init_printing()


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

        if not self.USE_FAST:  # USE Fast is defined in config
            # Get sympy expressions to compute dynamics
            self.get_combined_dynamics_kinematics()
            # self.get_inverse_dynamics()
            self.get_jacobians()
        else:
            # Get numba expressions for the dynamics
            self.calc_dqdt = fast_dqdt
            self.calc_model_jacobian_state = fast_model_jacobian_state
            self.calc_model_jacobian_input = fast_model_jacobian_input

        # Get expressions for the wheels dynamics
        self.get_wheels_dynamics()

    def _make_simbols(self):
        """
            Create sympy symbols
        """
        # state variables
        x, y, theta, thetadot = symbols("x, y, theta, thetadot", real=True)

        # static variables
        L, R, m, m_w, d = symbols("L, R, m, m_w, d", real=True)

        # control variables
        tau_r, tau_l = symbols("tau_r, tau_l", real=True)

        # speeds
        v, omega = symbols("v, omega", real=True)
        vdot, omegadot = symbols("vdot, omegadot", real=True)

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
        )

    def get_combined_dynamics_kinematics(self):
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
        ) = self.variables.values()

        # Define moments of inertia
        I_c = m * d ** 2  # mom. inertia around center of gravity
        I_w = m_w * R ** 2  # mom. inertia of wheels
        I = I_c + m * d ** 2 + 2 * m_w * L ** 2 + I_w

        # Define a constant:
        J = I + (2 * I ** 2 / R ** 2) * I_w

        # Define g vector and input vector
        g = Matrix([0, 0, 0, d * omega ** 2, -(m * d * omega * v) / J])
        inp = Matrix([v, omega, tau_r, tau_l])

        # Define M matrix
        M = Matrix(
            [
                [cos(theta), 0, 0, 0],
                [sin(theta), 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, L / (m * R), L / (m * R)],
                [0, 0, L / (J * R), -L / (J * R)],
            ]
        )

        # vectorize expression
        args = [theta, v, omega, L, R, m, d, m_w, tau_l, tau_r]
        expr = g + M * inp
        self.calc_dqdt = lambdify(args, expr, modules="numpy")

        # store matrices
        self.matrixes = dict(g=g, inp=inp, M=M,)

        # Store dxdt model as sympy expression
        self.model = g + M * inp

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
        ) = self.variables.values()

        # Get jacobian wrt state
        self.model_jacobian_state = self.model.jacobian(
            [x, y, theta, v, omega]
        )

        # Get jacobian wrt input
        self.model_jacobian_input = self.model.jacobian([tau_r, tau_l])

        # vectorize expressions
        args = [theta, v, omega, L, R, m, d, m_w]
        self.calc_model_jacobian_state = lambdify(
            args, self.model_jacobian_state, modules="numpy"
        )

        args = [L, R, m, d, m_w]

        self.calc_model_jacobian_input = lambdify(
            args, self.model_jacobian_input, modules="numpy"
        )

    def get_wheels_dynamics(self):
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
        ) = self.variables.values()
        """
            In the model you can use the wheels angular velocity
            to get the x,y,theta velocity.
            Here we do the inverse, given x,y,theta velocities
            we get the wheel's angular velocity.
            
            Using eqs 15 and 16 from the paper
        """

        nu_l_dot, nu_r_dot = symbols("nudot_L, nudot_R")

        # # define vecs and matrices
        # K = Matrix(
        #     [
        #         [R / 2 * cos(theta), R / 2 * cos(theta)],
        #         [R / 2 * sin(theta), R / 2 * sin(theta)],
        #         [R / (2 * L), -R / (2 * L)],
        #     ]
        # )

        # Q = Matrix([[sin(theta), 0], [cos(theta), 0], [0, 1]])

        # nu = K.pinv() * vels # * Q * vels
        args = [L, R, v, omega]
        vels = Matrix([v, omega])
        K = Matrix([[1 / R, 1 / R], [1 / R, -1 / R]])
        nu = K * vels
        self.calc_wheels_ang_vels = lambdify(args, nu, modules="numpy")

    def get_inverse_dynamics(self):
        """
            If the model is
                x_dot = g + M*tau
            the inverse model is
                tau = M_inv * (x_dot - g)
        """
        # Get variables
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
        ) = self.variables.values()
        state = Matrix([x, y, theta, v, omega])

        # Get inverse of M matrix
        M_inv = SparseMatrix(
            self.matrixes["M"]
        ).pinv()  # recast as sparse for speed

        # Get inverse model
        self.model_inverse = M_inv * (state - self.matrixes["g"])

        # Vectorize expression
        args = [x, y, theta, v, omega, L, R, m, d, m_w]

        self.calc_inv_dynamics = lambdify(
            args, self.model_inverse, modules="numpy"
        )
