import numpy as np
import scipy.linalg


from .config import iLQR_CONFIG, CONTROL_CONFIG, PLANNING_CONFIG
from ._cost import Cost


class Controller(Cost):
    def __init__(self, model):
        Cost.__init__(self)
        self.model = model

        self.pred_len = PLANNING_CONFIG["prediction_length"]

        self.input_size = CONTROL_CONFIG["INPUT_SIZE"]
        self.state_size = CONTROL_CONFIG["STATE_SIZE"]
        self.Q = CONTROL_CONFIG["Q"]
        self.R = CONTROL_CONFIG["R"]
        self.W = CONTROL_CONFIG["W"]
        self.angle_idx = CONTROL_CONFIG["ANGLE_IDX"]

        # store diagonal matrices
        self.Q_ = np.diag(self.Q)
        self.R_ = np.diag(self.R)
        self.W_ = np.diag(self.W)

        # Params
        self.max_iter = iLQR_CONFIG["max_iter"]
        self.init_mu = iLQR_CONFIG["init_mu"]
        self.mu = iLQR_CONFIG["init_mu"]
        self.mu_min = iLQR_CONFIG["mu_min"]
        self.mu_max = iLQR_CONFIG["mu_max"]
        self.init_delta = iLQR_CONFIG["init_delta"]
        self.delta = iLQR_CONFIG["init_delta"]
        self.threshold = iLQR_CONFIG["threshold"]

        # Initialize
        self.prev_sol = np.zeros(self.input_size)

    def linearize_with_finite_differences(self, x):
        """
            Use finite diferences to linearize the models dynamics
            see: https://studywolf.wordpress.com/2015/11/10/linear-quadratic-regulation-for-non-linear-systems-using-finite-differences/

            Arguments:
                x: Nx1 np.array with current state

            Returns:
                A: matrix of linearized spontaneous dynamics
                B: matrix of linearized input drive dynamics
        """
        delta = 1e-4
        n_inputs, n_states = self.input_size, self.state_size

        A = np.zeros((n_states, n_states))  # n states by n states
        B = np.zeros((n_states, n_inputs))  # n states by n inputs

        # get A
        for ii in range(n_states):
            _x = x.copy()

            # compute step forward
            _x[ii] += delta
            forward = self.model._fake_step(_x, self.prev_sol)

            # compute step backward
            _x[ii] -= 2 * delta
            backward = self.model._fake_step(_x, self.prev_sol)

            # store the average
            A[:, ii] = (forward - backward) / (2 * delta)

        # get B
        for ii in range(n_inputs):
            _u = self.prev_sol.copy()

            # compute step forward
            _u[ii] += delta
            forward = self.model._fake_step(x, _u)

            # compute step backward
            _u[ii] -= 2 * delta
            backward = self.model._fake_step(x, _u)

            # store the average
            B[:, ii] = (forward - backward) / (2 * delta)

        return A, B

    def dlqr(self, A, B):
        """Solve the discrete time lqr controller.
        
        
        x[k+1] = A x[k] + B u[k]
    
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]

        Arguments:
            A, B: np.arrays with linearized dynamics
        """
        Q, R = self.Q, self.R

        # first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

        return K

    def solve(self, curr_state, goal_states):
        """
            Solves the control problem using finite differences LQR
            see: https://studywolf.wordpress.com/2015/11/10/linear-quadratic-regulation-for-non-linear-systems-using-finite-differences/
        """
        # linearize
        A, B = self.linearize_with_finite_differences(curr_state)

        # solve LQR
        K = self.dlqr(A, B)

        # compute control
        state_error = self.fit_diff_in_range(curr_state - goal_states[0, :])
        u = -K @ state_error

        u = np.array(u).ravel()
        self.prev_sol = u
        return u

    # def calc_cost(self, curr_x, samples, g_xs):
    #     """ calculate the cost of input samples

    #     Args:
    #         curr_x (numpy.ndarray): shape(state_size),
    #             current robot position
    #         samples (numpy.ndarray): shape(pop_size, opt_dim),
    #             input samples
    #         g_xs (numpy.ndarray): shape(pred_len, state_size),
    #             goal states
    #     Returns:
    #         costs (numpy.ndarray): shape(pop_size, )
    #     """
    #     # get size
    #     pop_size = samples.shape[0]
    #     g_xs = np.tile(g_xs, (pop_size, 1, 1))

    #     # calc cost, pred_xs.shape = (pop_size, pred_len+1, state_size)
    #     pred_xs = self.model.predict_trajectory(curr_x, samples)

    #     # get particle cost
    #     costs = calc_cost(
    #         pred_xs, samples, g_xs, self.state_cost_fn, self.input_cost_fn,
    #     )

    #     return costs

    # def obtain_sol(self, curr_x, g_xs):
    #     """ calculate the optimal inputs

    #     Args:
    #         curr_x (numpy.ndarray): current state, shape(state_size, )
    #         g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
    #     Returns:
    #         opt_input (numpy.ndarray): optimal input, shape(input_size, )
    #     """
    #     # initialize
    #     sol = self.prev_sol.copy()
    #     update_sol = True
    #     self.mu = self.init_mu
    #     self.delta = self.init_delta

    #     # line search param
    #     alphas = 1.1 ** (-np.arange(10) ** 2)

    #     for opt_count in range(self.max_iter):
    #         accepted_sol = False

    #         # forward
    #         if update_sol:
    #             (
    #                 pred_xs,
    #                 cost,
    #                 f_x,
    #                 f_u,
    #                 l_x,
    #                 l_xx,
    #                 l_u,
    #                 l_uu,
    #                 l_ux,
    #             ) = self.forward(curr_x, g_xs, sol)
    #             update_sol = False

    #         try:
    #             # backward
    #             k, K = self.backward(f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux)

    #             # line search
    #             for alpha in alphas:
    #                 new_pred_xs, new_sol = self.calc_input(
    #                     k, K, pred_xs, sol, alpha
    #                 )

    #                 new_cost = calc_cost(
    #                     new_pred_xs[np.newaxis, :, :],
    #                     new_sol[np.newaxis, :, :],
    #                     g_xs[np.newaxis, :, :],
    #                     self.state_cost_fn,
    #                     self.input_cost_fn,
    #                 )

    #                 if new_cost < cost:
    #                     cost = new_cost
    #                     pred_xs = new_pred_xs
    #                     sol = new_sol
    #                     update_sol = True

    #                     # decrease regularization term
    #                     self.delta = min(1.0, self.delta) / self.init_delta
    #                     self.mu *= self.delta
    #                     if self.mu <= self.mu_min:
    #                         self.mu = 0.0

    #                     # accept the solution
    #                     accepted_sol = True
    #                     break

    #         except np.linalg.LinAlgError as e:
    #             print("Non ans : {}".format(e))

    #         if not accepted_sol:
    #             self.delta = max(1.0, self.delta) * self.init_delta
    #             self.mu = max(self.mu_min, self.mu * self.delta)
    #             if self.mu >= self.mu_max:
    #                 # print("Reach Max regularization term")
    #                 break

    #     # if not accepted_sol:
    #     #     print('Failed to converge to a solution')

    #     if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
    #         raise ValueError("nans or inf in solution!")

    #     # update prev sol
    #     self.prev_sol[:-1] = sol[1:]
    #     self.prev_sol[-1] = sol[-1]  # last use the terminal input

    #     return sol[0]

    # def calc_input(self, k, K, pred_xs, sol, alpha):
    #     """ calc input trajectory by using k and K

    #     Args:
    #         k (numpy.ndarray): gain, shape(pred_len, input_size)
    #         K (numpy.ndarray): gain, shape(pred_len, input_size, state_size)
    #         pred_xs (numpy.ndarray): predicted state,
    #             shape(pred_len+1, state_size)
    #         sol (numpy.ndarray): input trajectory, previous solutions
    #             shape(pred_len, input_size)
    #         alpha (float): param of line search
    #     Returns:
    #         new_pred_xs (numpy.ndarray): update state trajectory,
    #             shape(pred_len+1, state_size)
    #         new_sol (numpy.ndarray): update input trajectory,
    #             shape(pred_len, input_size)
    #     """

    #     # get size
    #     (pred_len, input_size, state_size) = K.shape

    #     # initialize
    #     new_pred_xs = np.zeros((pred_len + 1, state_size))
    #     new_pred_xs[0] = pred_xs[0].copy()  # init state is same
    #     new_sol = np.zeros((pred_len, input_size))

    #     for t in range(pred_len):
    #         new_sol[t] = (
    #             sol[t]
    #             + alpha * k[t]
    #             + np.dot(K[t], (new_pred_xs[t] - pred_xs[t]))
    #         )

    #         # if np.max(new_sol) > 100000:
    #         #     logger.warning(f'While computing imput, proposed solution is huge: {np.max(new_sol)}')

    #         new_pred_xs[t + 1] = self.model._fake_step(
    #             new_pred_xs[t], new_sol[t]
    #         )

    #     return new_pred_xs, new_sol

    # def forward(self, curr_x, g_xs, sol):
    #     """ forward step of iLQR

    #     Args:
    #         curr_x (numpy.ndarray): current state, shape(state_size, )
    #         g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
    #         sol (numpy.ndarray): solutions, shape(plan_len, input_size)
    #     Returns:
    #         f_x (numpy.ndarray): gradient of model with respecto to state,
    #             shape(pred_len, state_size, state_size)
    #         f_u (numpy.ndarray): gradient of model with respecto to input,
    #             shape(pred_len, state_size, input_size)
    #         l_x (numpy.ndarray): gradient of cost with respecto to state,
    #             shape(pred_len+1, state_size)
    #         l_u (numpy.ndarray): gradient of cost with respecto to input,
    #             shape(pred_len, input_size)
    #         l_xx (numpy.ndarray): hessian of cost with respecto to state,
    #             shape(pred_len+1, state_size, state_size)
    #         l_uu (numpy.ndarray): hessian of cost with respecto to input,
    #             shape(pred_len+1, input_size, input_size)
    #         l_ux (numpy.ndarray): hessian of cost with respect
    #             to state and input, shape(pred_len, input_size, state_size)
    #     """
    #     # simulate forward using the current control trajectory
    #     pred_xs = self.model.predict_trajectory(curr_x, sol)

    #     # check costs
    #     cost = self.calc_cost(curr_x, sol[np.newaxis, :, :], g_xs)

    #     # calc gradinet in batch
    #     f_x = self.model.calc_gradient(pred_xs[:-1], sol, wrt="x")
    #     f_u = self.model.calc_gradient(pred_xs[:-1], sol, wrt="u")

    #     # gradint of costs
    #     l_x, l_xx, l_u, l_uu, l_ux = self._calc_gradient_hessian_cost(
    #         pred_xs, g_xs, sol
    #     )

    #     return pred_xs, cost, f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux

    # def _calc_gradient_hessian_cost(self, pred_xs, g_x, sol):
    #     """ calculate gradient and hessian of model and cost fn

    #     Args:
    #         pred_xs (numpy.ndarray): predict traj,
    #             shape(pred_len+1, state_size)
    #         sol (numpy.ndarray): input traj,
    #             shape(pred_len, input_size)
    #     Returns
    #         l_x (numpy.ndarray): gradient of cost,
    #             shape(pred_len+1, state_size)
    #         l_u (numpy.ndarray): gradient of cost,
    #             shape(pred_len, input_size)
    #         l_xx (numpy.ndarray): hessian of cost,
    #             shape(pred_len+1, state_size, state_size)
    #         l_uu (numpy.ndarray): hessian of cost,
    #             shape(pred_len+1, input_size, input_size)
    #         l_ux (numpy.ndarray): hessian of cost,
    #             shape(pred_len, input_size, state_size)
    #     """
    #     # cost wrt to the state
    #     # l_x.shape = (pred_len+1, state_size)
    #     l_x = self.gradient_cost_fn_with_state(pred_xs[:-1], g_x[:-1])

    #     # cost wrt to the input
    #     # l_u.shape = (pred_len, input_size)
    #     l_u = self.gradient_cost_fn_with_input(pred_xs[:-1], sol)

    #     # l_xx.shape = (pred_len+1, state_size, state_size)
    #     l_xx = self.hessian_cost_fn_with_state(pred_xs[:-1], g_x[:-1])

    #     # l_uu.shape = (pred_len, input_size, input_size)
    #     l_uu = self.hessian_cost_fn_with_input(pred_xs[:-1], sol)

    #     # l_ux.shape = (pred_len, input_size, state_size)
    #     l_ux = self.hessian_cost_fn_with_input_state(pred_xs[:-1], sol)

    #     return l_x, l_xx, l_u, l_uu, l_ux

    # def backward(self, f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux):
    #     """ backward step of iLQR
    #     Args:
    #         f_x (numpy.ndarray): gradient of model with respecto to state,
    #             shape(pred_len+1, state_size, state_size)
    #         f_u (numpy.ndarray): gradient of model with respecto to input,
    #             shape(pred_len, state_size, input_size)
    #         l_x (numpy.ndarray): gradient of cost with respecto to state,
    #             shape(pred_len+1, state_size)
    #         l_u (numpy.ndarray): gradient of cost with respecto to input,
    #             shape(pred_len, input_size)
    #         l_xx (numpy.ndarray): hessian of cost with respecto to state,
    #             shape(pred_len+1, state_size, state_size)
    #         l_uu (numpy.ndarray): hessian of cost with respecto to input,
    #             shape(pred_len, input_size, input_size)
    #         l_ux (numpy.ndarray): hessian of cost with respect
    #             to state and input, shape(pred_len, input_size, state_size)

    #     Returns:
    #         k (numpy.ndarray): gain, shape(pred_len, input_size)
    #         K (numpy.ndarray): gain, shape(pred_len, input_size, state_size)
    #     """
    #     # get size
    #     (_, state_size, _) = f_x.shape

    #     # initialzie
    #     V_x = l_x[-1]
    #     V_xx = l_xx[-1]
    #     k = np.zeros((self.pred_len, self.input_size))
    #     K = np.zeros((self.pred_len, self.input_size, state_size))

    #     for t in range(self.pred_len - 1, -1, -1):
    #         # get Q val
    #         Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(
    #             f_x[t],
    #             f_u[t],
    #             l_x[t],
    #             l_u[t],
    #             l_xx[t],
    #             l_ux[t],
    #             l_uu[t],
    #             V_x,
    #             V_xx,
    #         )
    #         # calc gain
    #         k[t] = -np.linalg.solve(Q_uu, Q_u)
    #         K[t] = -np.linalg.solve(Q_uu, Q_ux)
    #         # update V_x val
    #         V_x = Q_x + np.dot(np.dot(K[t].T, Q_uu), k[t])
    #         V_x += np.dot(K[t].T, Q_u) + np.dot(Q_ux.T, k[t])
    #         # update V_xx val
    #         V_xx = Q_xx + np.dot(np.dot(K[t].T, Q_uu), K[t])
    #         V_xx += np.dot(K[t].T, Q_ux) + np.dot(Q_ux.T, K[t])
    #         V_xx = 0.5 * (V_xx + V_xx.T)  # to maintain symmetry.

    #     return k, K

    # def _Q(self, f_x, f_u, l_x, l_u, l_xx, l_ux, l_uu, V_x, V_xx):
    #     """Computes second order expansion.
    #     Args:
    #         f_x (numpy.ndarray): gradient of model with respecto to state,
    #             shape(state_size, state_size)
    #         f_u (numpy.ndarray): gradient of model with respecto to input,
    #             shape(state_size, input_size)
    #         l_x (numpy.ndarray): gradient of cost with respecto to state,
    #             shape(state_size, )
    #         l_u (numpy.ndarray): gradient of cost with respecto to input,
    #             shape(input_size, )
    #         l_xx (numpy.ndarray): hessian of cost with respecto to state,
    #             shape(state_size, state_size)
    #         l_uu (numpy.ndarray): hessian of cost with respecto to input,
    #             shape(input_size, input_size)
    #         l_ux (numpy.ndarray): hessian of cost with respect
    #             to state and input, shape(input_size, state_size)
    #         V_x (numpy.ndarray): gradient of Value function,
    #             shape(state_size, )
    #         V_xx (numpy.ndarray): hessian of Value function,
    #             shape(state_size, state_size)
    #     Returns:
    #         Q_x (numpy.ndarray): gradient of Q function, shape(state_size, )
    #         Q_u (numpy.ndarray): gradient of Q function, shape(input_size, )
    #         Q_xx (numpy.ndarray): hessian of Q fucntion,
    #             shape(state_size, state_size)
    #         Q_ux (numpy.ndarray): hessian of Q fucntion,
    #             shape(input_size, state_size)
    #         Q_uu (numpy.ndarray): hessian of Q fucntion,
    #             shape(input_size, input_size)
    #     """
    #     # get size
    #     state_size = len(l_x)

    #     Q_x = l_x + np.dot(f_x.T, V_x)
    #     Q_u = l_u + np.dot(f_u.T, V_x)
    #     Q_xx = l_xx + np.dot(np.dot(f_x.T, V_xx), f_x)

    #     reg = self.mu * np.eye(state_size)
    #     Q_ux = l_ux + np.dot(np.dot(f_u.T, (V_xx + reg)), f_x)
    #     Q_uu = l_uu + np.dot(np.dot(f_u.T, (V_xx + reg)), f_u)

    #     return Q_x, Q_u, Q_xx, Q_ux, Q_uu
