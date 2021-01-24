import numpy as np

# from loguru import logger

from control import config
from ._cost import Cost, calc_cost


class Controller(Cost):
    def __init__(self, model):
        Cost.__init__(self)
        self.model = model

        self.controls_size = config.CONTROL_CONFIG["controls_size"]
        self.state_size = config.CONTROL_CONFIG["STATE_SIZE"]
        self.angle_idx = config.CONTROL_CONFIG["ANGLE_IDX"]

        # Params
        self.max_iter = config.iLQR_CONFIG["max_iter"]
        self.init_mu = config.iLQR_CONFIG["init_mu"]
        self.mu = config.iLQR_CONFIG["init_mu"]
        self.mu_min = config.iLQR_CONFIG["mu_min"]
        self.mu_max = config.iLQR_CONFIG["mu_max"]
        self.init_delta = config.iLQR_CONFIG["init_delta"]
        self.delta = config.iLQR_CONFIG["init_delta"]
        self.threshold = config.iLQR_CONFIG["threshold"]

        # Initialize
        self.prev_sol = None

    def update_matrices(self):
        # get matrices
        self.Q = config.CONTROL_CONFIG["Q"]
        self.R = config.CONTROL_CONFIG["R"]
        self.W = config.CONTROL_CONFIG["W"]
        self.Z = config.CONTROL_CONFIG["Z"]

        # store diags to speed up computations
        self.Q_ = np.diag(self.Q)
        self.R_ = np.diag(self.R)
        self.W_ = np.diag(self.W)
        self.Z_ = np.diag(self.Z)

    def solve(self, X, X_g):
        """ calculate the optimal inputs

        Args:
            X (numpy.ndarray): current state, shape(state_size, )
            X_g (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(controls_size, )
        """
        self.pred_len = config.PLANNING_CONFIG["prediction_length"]
        self.update_matrices()

        # get previous solution and adjust to adapt to prediction length

        if self.prev_sol is None:
            self.prev_sol = np.zeros((self.pred_len, self.controls_size))

        if len(self.prev_sol) == self.pred_len:
            U = self.prev_sol.copy()
        else:
            U = np.zeros((self.pred_len, self.controls_size))
            U[: len(self.prev_sol)] = self.prev_sol
            U[len(self.prev_sol) :] = self.prev_sol[-1]  # noqa: E203
            self.prev_sol = U

        # initialize variables
        update_sol = True
        self.mu = self.init_mu
        self.delta = self.init_delta

        # line search param
        alphas = 1.1 ** (-np.arange(10) ** 2)

        for opt_count in range(self.max_iter):
            accepted_sol = False

            # forward
            if update_sol:
                (
                    pred_xs,  # predicted X traj. n steps x n states
                    cost,  # cost of predicted traj, float
                    f_x,  # pred len x n states x n states
                    f_u,  # pred len x n states x n inputs
                    l_x,  # pred len x n states
                    l_xx,  # pred len x n states x n states
                    l_u,  # pred len x n inputs
                    l_uu,  # pred len x n inputs x n inputs
                    l_ux,  # pred len x n inputs x n states
                ) = self.forward(X, X_g, U)
                update_sol = False

            # backward
            k, K = self.backward(f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux)

            # line search for best solution
            for alpha in alphas:
                X_hat, U_hat = self.calc_input(k, K, pred_xs, U, alpha)

                new_cost = calc_cost(
                    X_hat[np.newaxis, :, :],
                    U_hat[np.newaxis, :, :],
                    self.prev_sol[np.newaxis, :, :],
                    X_g[np.newaxis, :, :],
                    self.state_cost_fn,
                    self.input_cost_fn,
                )

                if new_cost < cost:
                    cost = new_cost
                    pred_xs = X_hat
                    U = U_hat
                    update_sol = True

                    # decrease regularization term
                    self.delta = min(1.0, self.delta) / self.init_delta
                    self.mu *= self.delta
                    if self.mu <= self.mu_min:
                        self.mu = 0.0

                    # accept the solution
                    accepted_sol = True
                    break

            if not accepted_sol:
                # adjust params
                self.delta = max(1.0, self.delta) * self.init_delta
                self.mu = max(self.mu_min, self.mu * self.delta)
                if self.mu >= self.mu_max:
                    # logger.debug("Reached max regularization term")
                    break

        # if not accepted_sol:
        #     logger.debug("Failed to converge to a solution")

        if np.any(np.isnan(U)) or np.any(np.isinf(U)):
            raise ValueError("nans or inf in solution!")

        # update prev U
        self.prev_sol[:-1] = U[1:]
        self.prev_sol[-1] = U[-1]  # last use the terminal input
        return U[0]

    def calc_input(self, k, K, pred_xs, U, alpha):
        """ calc input trajectory by using k and K

        Args:
            k (numpy.ndarray): gain, shape(pred_len, controls_size)
            K (numpy.ndarray): gain, shape(pred_len, controls_size, state_size)
            pred_xs (numpy.ndarray): predicted state,
                shape(pred_len+1, state_size)
            U (numpy.ndarray): input trajectory, previous solutions
                shape(pred_len, controls_size)
            alpha (float): param of line search
        Returns:
            X_hat (numpy.ndarray): update state trajectory,
                shape(pred_len+1, state_size)
            U_hat (numpy.ndarray): update input trajectory,
                shape(pred_len, controls_size)
        """

        # get size
        (pred_len, controls_size, state_size) = K.shape

        # initialize
        X_hat = np.zeros((pred_len + 1, state_size))
        X_hat[0] = pred_xs[0].copy()  # init state is same
        U_hat = np.zeros((pred_len, controls_size))

        for t in range(pred_len):
            U_hat[t] = (
                U[t] + alpha * k[t] + np.dot(K[t], (X_hat[t] - pred_xs[t]))
            )

            try:
                X_hat[t + 1] = self.model._fake_step(X_hat[t], U_hat[t])
            except ValueError:
                raise ValueError(
                    "Failed to update controls with iLQR, "
                    + "likely nans or infs came up"
                )

        return X_hat, U_hat

    def forward(self, curr_x, X_g, U):
        """ forward step of iLQR

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            X_g (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
            U (numpy.ndarray): solutions, shape(plan_len, controls_size)

        Returns:
            f_x (numpy.ndarray): gradient of model with respecto to state,
                shape(pred_len, state_size, state_size)
            f_u (numpy.ndarray): gradient of model with respecto to input,
                shape(pred_len, state_size, controls_size)
            l_x (numpy.ndarray): gradient of cost with respecto to state,
                shape(pred_len+1, state_size)
            l_u (numpy.ndarray): gradient of cost with respecto to input,
                shape(pred_len, controls_size)
            l_xx (numpy.ndarray): hessian of cost with respecto to state,
                shape(pred_len+1, state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost with respecto to input,
                shape(pred_len+1, controls_size, controls_size)
            l_ux (numpy.ndarray): hessian of cost with respect
                to state and input, shape(pred_len, controls_size, state_size)
        """
        # simulate forward using the current control trajectory
        X = self.model.predict_trajectory(curr_x, U)

        # check costs
        cost = self.calc_cost(
            curr_x, U[np.newaxis, :, :], self.prev_sol[np.newaxis, :, :], X_g
        )

        # calc gradient in batch
        f_x = self.model.calc_gradient(X[:-1], U, wrt="x")
        f_u = self.model.calc_gradient(X[:-1], U, wrt="u")

        # gradint of costs
        l_x, l_xx, l_u, l_uu, l_ux = self._calc_gradient_hessian_cost(
            X, X_g, U
        )

        for var in (l_x, l_xx, l_u, l_uu, l_ux):
            if np.any(np.isnan(var)) or np.any(np.isinf(var)):
                raise ValueError("Got nans while running iLQR forward!")

        return X, cost, f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux

    def _calc_gradient_hessian_cost(self, X, g_x, U):
        """ calculate gradient and hessian of model and cost fn

        Args:
            X (numpy.ndarray): predict traj,
                shape(pred_len+1, state_size)
            U (numpy.ndarray): input traj,
                shape(pred_len, controls_size)
        Returns
            l_x (numpy.ndarray): gradient of cost,
                shape(pred_len+1, state_size)
            l_u (numpy.ndarray): gradient of cost,
                shape(pred_len, controls_size)
            l_xx (numpy.ndarray): hessian of cost,
                shape(pred_len+1, state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost,
                shape(pred_len+1, controls_size, controls_size)
            l_ux (numpy.ndarray): hessian of cost,
                shape(pred_len, controls_size, state_size)
        """
        # cost wrt to the state
        # l_x.shape = (pred_len+1, state_size)
        l_x = (
            self.gradient_cost_fn_with_state(X[:-1], g_x[:-1])
            * config.PARAMS["dt"]
        )

        # cost wrt to the input
        # l_u.shape = (pred_len, controls_size)
        l_u = (
            self.gradient_cost_fn_with_input(X[:-1], U, self.prev_sol)
            * config.PARAMS["dt"]
        )

        # l_xx.shape = (pred_len+1, state_size, state_size)
        l_xx = (
            self.hessian_cost_fn_with_state(X[:-1], g_x[:-1])
            * config.PARAMS["dt"]
        )

        # l_uu.shape = (pred_len, controls_size, controls_size)
        l_uu = self.hessian_cost_fn_with_input(X[:-1], U) * config.PARAMS["dt"]

        # l_ux.shape = (pred_len, controls_size, state_size)
        l_ux = (
            self.hessian_cost_fn_with_input_state(X[:-1], U)
            * config.PARAMS["dt"]
        )

        return l_x, l_xx, l_u, l_uu, l_ux

    def backward(self, f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux):
        """ backward step of iLQR
        Args:
            f_x (numpy.ndarray): gradient of model with respecto to state,
                shape(pred_len+1, state_size, state_size)
            f_u (numpy.ndarray): gradient of model with respecto to input,
                shape(pred_len, state_size, controls_size)
            l_x (numpy.ndarray): gradient of cost with respecto to state,
                shape(pred_len+1, state_size)
            l_u (numpy.ndarray): gradient of cost with respecto to input,
                shape(pred_len, controls_size)
            l_xx (numpy.ndarray): hessian of cost with respecto to state,
                shape(pred_len+1, state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost with respecto to input,
                shape(pred_len, controls_size, controls_size)
            l_ux (numpy.ndarray): hessian of cost with respect
                to state and input, shape(pred_len, controls_size, state_size)

        Returns:
            k (numpy.ndarray): gain, shape(pred_len, controls_size)
            K (numpy.ndarray): gain, shape(pred_len, controls_size, state_size)
        """
        # get size
        (_, state_size, _) = f_x.shape

        # initialzie
        V_x = l_x[-1].copy()
        V_xx = l_xx[-1].copy()
        k = np.zeros((self.pred_len, self.controls_size))
        K = np.zeros((self.pred_len, self.controls_size, state_size))

        for t in range(self.pred_len - 1, -1, -1):
            # get Q val
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(
                f_x[t],
                f_u[t],
                l_x[t],
                l_u[t],
                l_xx[t],
                l_ux[t],
                l_uu[t],
                V_x,
                V_xx,
            )

            # invert and regularize Q_uu
            Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
            Q_uu_evals[Q_uu_evals < 0] = 0.0
            Q_uu_evals += self.mu
            Q_uu_inv = np.dot(
                Q_uu_evecs, np.dot(np.diag(1.0 / Q_uu_evals), Q_uu_evecs.T)
            )
            # calc gain
            k[t] = -Q_uu_inv.dot(Q_u)
            K[t] = -Q_uu_inv.dot(Q_ux)

            # update V_x val
            V_x = Q_x + np.dot(np.dot(K[t].T, Q_uu), k[t])
            V_x += np.dot(K[t].T, Q_u) + np.dot(Q_ux.T, k[t])

            # update V_xx val
            V_xx = Q_xx + np.dot(np.dot(K[t].T, Q_uu), K[t])
            V_xx += np.dot(K[t].T, Q_ux) + np.dot(Q_ux.T, K[t])
            V_xx = 0.5 * (V_xx + V_xx.T)  # to maintain symmetry.

        if np.any(np.isnan(k)) or np.any(np.isnan(K)):
            raise ValueError("nans came up during backward pass of iLQR")

        return k, K

    def _Q(self, f_x, f_u, l_x, l_u, l_xx, l_ux, l_uu, V_x, V_xx):
        """Computes second order expansion.
        Args:
            f_x (numpy.ndarray): gradient of model with respecto to state,
                shape(state_size, state_size)
            f_u (numpy.ndarray): gradient of model with respecto to input,
                shape(state_size, controls_size)
            l_x (numpy.ndarray): gradient of cost with respecto to state,
                shape(state_size, )
            l_u (numpy.ndarray): gradient of cost with respecto to input,
                shape(controls_size, )
            l_xx (numpy.ndarray): hessian of cost with respecto to state,
                shape(state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost with respecto to input,
                shape(controls_size, controls_size)
            l_ux (numpy.ndarray): hessian of cost with respect
                to state and input, shape(controls_size, state_size)
            V_x (numpy.ndarray): gradient of Value function,
                shape(state_size, )
            V_xx (numpy.ndarray): hessian of Value function,
                shape(state_size, state_size)
        Returns:
            Q_x (numpy.ndarray): gradient of Q function, shape(state_size, )
            Q_u (numpy.ndarray): gradient of Q function, shape(controls_size, )
            Q_xx (numpy.ndarray): hessian of Q fucntion,
                shape(state_size, state_size)
            Q_ux (numpy.ndarray): hessian of Q fucntion,
                shape(controls_size, state_size)
            Q_uu (numpy.ndarray): hessian of Q fucntion,
                shape(controls_size, controls_size)
        """
        # get size
        # state_size = len(l_x)

        Q_x = l_x + np.dot(f_x.T, V_x)
        Q_u = l_u + np.dot(f_u.T, V_x)
        Q_xx = l_xx + np.dot(np.dot(f_x.T, V_xx), f_x)

        # reg = self.mu * np.eye(state_size)
        # Q_ux = l_ux + np.dot(np.dot(f_u.T, (V_xx + reg)), f_x)
        # Q_uu = l_uu + np.dot(np.dot(f_u.T, (V_xx + reg)), f_u)

        Q_ux = l_ux + np.dot(np.dot(f_u.T, (V_xx)), f_x)
        Q_uu = l_uu + np.dot(np.dot(f_u.T, (V_xx)), f_u)
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
