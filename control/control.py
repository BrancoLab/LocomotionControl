import numpy as np

from .config import iLQR_CONFIG, CONTROL_CONFIG, PLANNING_CONFIG


def fit_angle_in_range(
    angles, min_angle=-np.pi, max_angle=np.pi, is_deg=False
):
    """ Check angle range and correct the range
    it assumes that the angles are passed ind degrees
    
    Args:
        angle (numpy.ndarray): in radians
        min_angle (float): maximum of range in radians, default -pi
        max_angle (float): minimum of range in radians, default pi
    Returns: 
        fitted_angle (numpy.ndarray): range angle in radians
    """
    if max_angle < min_angle:
        raise ValueError("max angle must be greater than min angle")
    if (max_angle - min_angle) < 2.0 * np.pi:
        raise ValueError(
            "difference between max_angle \
                          and min_angle must be greater than 2.0 * pi"
        )
    if is_deg:
        output = np.radians(angles)
    else:
        output = np.array(angles)

    output_shape = output.shape

    output = output.flatten()
    output -= min_angle
    output %= 2 * np.pi
    output += 2 * np.pi
    output %= 2 * np.pi
    output += min_angle

    output = np.minimum(max_angle, np.maximum(min_angle, output))
    output = output.reshape(output_shape)

    # if is_deg:
    #     output = np.degrees(output)
    return output


def calc_cost(
    pred_xs,
    input_sample,
    g_xs,
    state_cost_fn,
    input_cost_fn,
    terminal_state_cost_fn,
):
    """ calculate the cost 

    Args:
        pred_xs (numpy.ndarray): predicted state trajectory, 
            shape(pop_size, pred_len+1, state_size)
        input_sample (numpy.ndarray): inputs samples trajectory,
            shape(pop_size, pred_len+1, input_size)
        g_xs (numpy.ndarray): goal state trajectory,
            shape(pop_size, pred_len+1, state_size)
        state_cost_fn (function): state cost fucntion
        input_cost_fn (function): input cost fucntion
        terminal_state_cost_fn (function): terminal state cost fucntion
    Returns:
        cost (numpy.ndarray): cost of the input sample, shape(pop_size, )
    """
    # state cost
    state_cost = 0.0
    if state_cost_fn is not None:
        state_pred_par_cost = state_cost_fn(
            pred_xs[:, 1:-1, :], g_xs[:, 1:-1, :]
        )
        state_cost = np.sum(np.sum(state_pred_par_cost, axis=-1), axis=-1)

    # terminal cost
    terminal_state_cost = 0.0
    if terminal_state_cost_fn is not None:
        terminal_state_par_cost = terminal_state_cost_fn(
            pred_xs[:, -1, :], g_xs[:, -1, :]
        )
        terminal_state_cost = np.sum(terminal_state_par_cost, axis=-1)

    # act cost
    act_cost = 0.0
    if input_cost_fn is not None:
        act_pred_par_cost = input_cost_fn(input_sample)
        act_cost = np.sum(np.sum(act_pred_par_cost, axis=-1), axis=-1)

    return state_cost + terminal_state_cost + act_cost


class Cost:
    def fit_diff_in_range(self, diff_x):
        """ fit difference state in range(angle)

        Args:
            diff_x (numpy.ndarray): 
                shape(pop_size, pred_len, state_size) or
                shape(pred_len, state_size) or
                shape(state_size, )
        Returns:
            fitted_diff_x (numpy.ndarray): same shape as diff_x
        """

        if len(diff_x.shape) == 3:
            diff_x[:, :, self.angle_idx] = fit_angle_in_range(
                diff_x[:, :, self.angle_idx]
            )
        elif len(diff_x.shape) == 2:
            diff_x[:, self.angle_idx] = fit_angle_in_range(
                diff_x[:, self.angle_idx]
            )
        elif len(diff_x.shape) == 1:
            diff_x[self.angle_idx] = fit_angle_in_range(diff_x[self.angle_idx])

        return diff_x

    def input_cost_fn(self, u):
        """ input cost functions
        Args:
            u (numpy.ndarray): input, shape(pred_len, input_size)
                or shape(pop_size, pred_len, input_size)
        Returns:
            cost (numpy.ndarray): cost of input, shape(pred_len, input_size) or
                shape(pop_size, pred_len, input_size)
        """
        return (u ** 2) * np.diag(self.R)

    def state_cost_fn(self, x, g_x):
        """ state cost function
        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, shape(pred_len, state_size) or
                shape(pop_size, pred_len, state_size)


            Cost of state, is given bY
            (x - X_g)T * Q * (x - x_g)
        """
        diff = self.fit_diff_in_range(x - g_x)
        # diff = x - g_x
        return (diff ** 2) * np.diag(self.Q)

    def terminal_state_cost_fn(self, terminal_x, terminal_g_x):
        """
        Args:
            terminal_x (numpy.ndarray): terminal state,
                shape(state_size, ) or shape(pop_size, state_size)
            terminal_g_x (numpy.ndarray): terminal goal state,
                shape(state_size, ) or shape(pop_size, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, shape(pred_len, ) or
                shape(pop_size, pred_len)

        Cost of end state, is given bY
            (x - X_g)T * Q * (x - x_g)
        """
        terminal_diff = self.fit_diff_in_range(terminal_x - terminal_g_x)
        # terminal_diff = terminal_x  - terminal_g_x
        return ((terminal_diff) ** 2) * np.diag(self.Sf)

    def calc_step_cost(self, x, u, g_x):
        cost = dict(
            control=self.model._control(*self.input_cost_fn(u)),
            state=self.model._state(*self.state_cost_fn(x, g_x)),
        )
        cost["total"] = (
            np.array(cost["control"]).sum() + np.array(cost["state"]).sum()
        )
        return cost

    def gradient_cost_fn_with_state(self, x, g_x, terminal=False):
        """ gradient of costs with respect to the state

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)
        
        Returns:
            l_x (numpy.ndarray): gradient of cost, shape(pred_len, state_size)
                or shape(1, state_size)
        """
        diff = self.fit_diff_in_range(x - g_x)

        if not terminal:
            return 2.0 * (diff) * np.diag(self.Q)

        return (2.0 * (diff) * np.diag(self.Sf))[np.newaxis, :]

    # ---------------------------- COST FUN GRADIENTS ---------------------------- #

    def gradient_cost_fn_with_input(self, x, u):
        """ gradient of costs with respect to the input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        
        Returns:
            l_u (numpy.ndarray): gradient of cost, shape(pred_len, input_size)
        """
        return 2.0 * u * np.diag(self.R)

    def hessian_cost_fn_with_state(self, x, g_x, terminal=False):
        """ hessian costs with respect to the state

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)
        
        Returns:
            l_xx (numpy.ndarray): gradient of cost,
                shape(pred_len, state_size, state_size) or
                shape(1, state_size, state_size) or
        """
        if not terminal:
            (pred_len, _) = x.shape
            return np.tile(2.0 * self.Q, (pred_len, 1, 1))

        return np.tile(2.0 * self.Sf, (1, 1, 1))

    def hessian_cost_fn_with_input(self, x, u):
        """ hessian costs with respect to the input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        
        Returns:
            l_uu (numpy.ndarray): gradient of cost,
                shape(pred_len, input_size, input_size)
        """
        (pred_len, _) = u.shape

        return np.tile(2.0 * self.R, (pred_len, 1, 1))

    def hessian_cost_fn_with_input_state(self, x, u):
        """ hessian costs with respect to the state and input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        
        Returns:
            l_ux (numpy.ndarray): gradient of cost ,
                shape(pred_len, input_size, state_size)
        """
        (_, state_size) = x.shape
        (pred_len, input_size) = u.shape

        return np.zeros((pred_len, input_size, state_size))


class Controller(Cost):
    def __init__(self, model):
        Cost.__init__(self)
        self.model = model

        self.pred_len = PLANNING_CONFIG["prediction_length"]

        self.input_size = CONTROL_CONFIG["INPUT_SIZE"]
        self.state_size = CONTROL_CONFIG["STATE_SIZE"]
        self.Q = CONTROL_CONFIG["Q"]
        self.R = CONTROL_CONFIG["R"]
        self.Sf = CONTROL_CONFIG["Sf"]
        self.angle_idx = CONTROL_CONFIG["ANGLE_IDX"]

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
        self.prev_sol = np.zeros((self.pred_len, self.input_size))

    def calc_cost(self, curr_x, samples, g_xs):
        """ calculate the cost of input samples

        Args:
            curr_x (numpy.ndarray): shape(state_size),
                current robot position
            samples (numpy.ndarray): shape(pop_size, opt_dim), 
                input samples
            g_xs (numpy.ndarray): shape(pred_len, state_size),
                goal states
        Returns:
            costs (numpy.ndarray): shape(pop_size, )
        """
        # get size
        pop_size = samples.shape[0]
        g_xs = np.tile(g_xs, (pop_size, 1, 1))

        # calc cost, pred_xs.shape = (pop_size, pred_len+1, state_size)
        pred_xs = self.model.predict_trajectory(curr_x, samples)

        # get particle cost
        costs = calc_cost(
            pred_xs,
            samples,
            g_xs,
            self.state_cost_fn,
            self.input_cost_fn,
            self.terminal_state_cost_fn,
        )

        return costs

    def obtain_sol(self, curr_x, g_xs):
        """ calculate the optimal inputs

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        # initialize
        sol = self.prev_sol.copy()
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
                    pred_xs,
                    cost,
                    f_x,
                    f_u,
                    l_x,
                    l_xx,
                    l_u,
                    l_uu,
                    l_ux,
                ) = self.forward(curr_x, g_xs, sol)
                update_sol = False

            try:
                # backward
                k, K = self.backward(f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux)

                # line search
                for alpha in alphas:
                    new_pred_xs, new_sol = self.calc_input(
                        k, K, pred_xs, sol, alpha
                    )

                    new_cost = calc_cost(
                        new_pred_xs[np.newaxis, :, :],
                        new_sol[np.newaxis, :, :],
                        g_xs[np.newaxis, :, :],
                        self.state_cost_fn,
                        self.input_cost_fn,
                        self.terminal_state_cost_fn,
                    )

                    if new_cost < cost:
                        cost = new_cost
                        pred_xs = new_pred_xs
                        sol = new_sol
                        update_sol = True

                        # decrease regularization term
                        self.delta = min(1.0, self.delta) / self.init_delta
                        self.mu *= self.delta
                        if self.mu <= self.mu_min:
                            self.mu = 0.0

                        # accept the solution
                        accepted_sol = True
                        break

            except np.linalg.LinAlgError as e:
                print("Non ans : {}".format(e))

            if not accepted_sol:
                self.delta = max(1.0, self.delta) * self.init_delta
                self.mu = max(self.mu_min, self.mu * self.delta)
                if self.mu >= self.mu_max:
                    # print("Reach Max regularization term")
                    break

        # if not accepted_sol:
        #     print('Failed to converge to a solution')

        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            raise ValueError("nans or inf in solution!")

        # update prev sol
        self.prev_sol[:-1] = sol[1:]
        self.prev_sol[-1] = sol[-1]  # last use the terminal input

        return sol[0]

    def calc_input(self, k, K, pred_xs, sol, alpha):
        """ calc input trajectory by using k and K

        Args:
            k (numpy.ndarray): gain, shape(pred_len, input_size)
            K (numpy.ndarray): gain, shape(pred_len, input_size, state_size)
            pred_xs (numpy.ndarray): predicted state,
                shape(pred_len+1, state_size)
            sol (numpy.ndarray): input trajectory, previous solutions
                shape(pred_len, input_size)
            alpha (float): param of line search
        Returns:
            new_pred_xs (numpy.ndarray): update state trajectory,
                shape(pred_len+1, state_size)
            new_sol (numpy.ndarray): update input trajectory,
                shape(pred_len, input_size)
        """

        # get size
        (pred_len, input_size, state_size) = K.shape

        # initialize
        new_pred_xs = np.zeros((pred_len + 1, state_size))
        new_pred_xs[0] = pred_xs[0].copy()  # init state is same
        new_sol = np.zeros((pred_len, input_size))

        for t in range(pred_len):
            new_sol[t] = (
                sol[t]
                + alpha * k[t]
                + np.dot(K[t], (new_pred_xs[t] - pred_xs[t]))
            )
            new_pred_xs[t + 1] = self.model._fake_step(
                new_pred_xs[t], new_sol[t]
            )

        return new_pred_xs, new_sol

    def forward(self, curr_x, g_xs, sol):
        """ forward step of iLQR

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
            sol (numpy.ndarray): solutions, shape(plan_len, input_size)
        Returns:
            f_x (numpy.ndarray): gradient of model with respecto to state,
                shape(pred_len, state_size, state_size)
            f_u (numpy.ndarray): gradient of model with respecto to input,
                shape(pred_len, state_size, input_size)
            l_x (numpy.ndarray): gradient of cost with respecto to state,
                shape(pred_len+1, state_size)
            l_u (numpy.ndarray): gradient of cost with respecto to input,
                shape(pred_len, input_size)
            l_xx (numpy.ndarray): hessian of cost with respecto to state,
                shape(pred_len+1, state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost with respecto to input,
                shape(pred_len+1, input_size, input_size)
            l_ux (numpy.ndarray): hessian of cost with respect
                to state and input, shape(pred_len, input_size, state_size)
        """
        # simulate forward using the current control trajectory
        pred_xs = self.model.predict_trajectory(curr_x, sol)

        # check costs
        cost = self.calc_cost(curr_x, sol[np.newaxis, :, :], g_xs)

        # calc gradinet in batch
        f_x = self.model.calc_gradient(pred_xs[:-1], sol, wrt="x")
        f_u = self.model.calc_gradient(pred_xs[:-1], sol, wrt="u")

        # gradint of costs
        l_x, l_xx, l_u, l_uu, l_ux = self._calc_gradient_hessian_cost(
            pred_xs, g_xs, sol
        )

        return pred_xs, cost, f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux

    def _calc_gradient_hessian_cost(self, pred_xs, g_x, sol):
        """ calculate gradient and hessian of model and cost fn
        
        Args:
            pred_xs (numpy.ndarray): predict traj,
                shape(pred_len+1, state_size)
            sol (numpy.ndarray): input traj,
                shape(pred_len, input_size)
        Returns
            l_x (numpy.ndarray): gradient of cost,
                shape(pred_len+1, state_size)
            l_u (numpy.ndarray): gradient of cost,
                shape(pred_len, input_size)
            l_xx (numpy.ndarray): hessian of cost,
                shape(pred_len+1, state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost,
                shape(pred_len+1, input_size, input_size)
            l_ux (numpy.ndarray): hessian of cost, 
                shape(pred_len, input_size, state_size)
        """
        # l_x.shape = (pred_len+1, state_size)
        l_x = self.gradient_cost_fn_with_state(
            pred_xs[:-1], g_x[:-1], terminal=False
        )
        terminal_l_x = self.gradient_cost_fn_with_state(
            pred_xs[-1], g_x[-1], terminal=True
        )

        l_x = np.concatenate((l_x, terminal_l_x), axis=0)

        # l_u.shape = (pred_len, input_size)
        l_u = self.gradient_cost_fn_with_input(pred_xs[:-1], sol)

        # l_xx.shape = (pred_len+1, state_size, state_size)
        l_xx = self.hessian_cost_fn_with_state(
            pred_xs[:-1], g_x[:-1], terminal=False
        )
        terminal_l_xx = self.hessian_cost_fn_with_state(
            pred_xs[-1], g_x[-1], terminal=True
        )

        l_xx = np.concatenate((l_xx, terminal_l_xx), axis=0)

        # l_uu.shape = (pred_len, input_size, input_size)
        l_uu = self.hessian_cost_fn_with_input(pred_xs[:-1], sol)

        # l_ux.shape = (pred_len, input_size, state_size)
        l_ux = self.hessian_cost_fn_with_input_state(pred_xs[:-1], sol)

        return l_x, l_xx, l_u, l_uu, l_ux

    def backward(self, f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux):
        """ backward step of iLQR
        Args:
            f_x (numpy.ndarray): gradient of model with respecto to state,
                shape(pred_len+1, state_size, state_size)
            f_u (numpy.ndarray): gradient of model with respecto to input,
                shape(pred_len, state_size, input_size)
            l_x (numpy.ndarray): gradient of cost with respecto to state,
                shape(pred_len+1, state_size)
            l_u (numpy.ndarray): gradient of cost with respecto to input,
                shape(pred_len, input_size)
            l_xx (numpy.ndarray): hessian of cost with respecto to state,
                shape(pred_len+1, state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost with respecto to input,
                shape(pred_len, input_size, input_size)
            l_ux (numpy.ndarray): hessian of cost with respect
                to state and input, shape(pred_len, input_size, state_size)
        
        Returns:
            k (numpy.ndarray): gain, shape(pred_len, input_size)
            K (numpy.ndarray): gain, shape(pred_len, input_size, state_size)
        """
        # get size
        (_, state_size, _) = f_x.shape

        # initialzie
        V_x = l_x[-1]
        V_xx = l_xx[-1]
        k = np.zeros((self.pred_len, self.input_size))
        K = np.zeros((self.pred_len, self.input_size, state_size))

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
            # calc gain
            k[t] = -np.linalg.solve(Q_uu, Q_u)
            K[t] = -np.linalg.solve(Q_uu, Q_ux)
            # update V_x val
            V_x = Q_x + np.dot(np.dot(K[t].T, Q_uu), k[t])
            V_x += np.dot(K[t].T, Q_u) + np.dot(Q_ux.T, k[t])
            # update V_xx val
            V_xx = Q_xx + np.dot(np.dot(K[t].T, Q_uu), K[t])
            V_xx += np.dot(K[t].T, Q_ux) + np.dot(Q_ux.T, K[t])
            V_xx = 0.5 * (V_xx + V_xx.T)  # to maintain symmetry.

        return k, K

    def _Q(self, f_x, f_u, l_x, l_u, l_xx, l_ux, l_uu, V_x, V_xx):
        """Computes second order expansion.
        Args:
            f_x (numpy.ndarray): gradient of model with respecto to state,
                shape(state_size, state_size)
            f_u (numpy.ndarray): gradient of model with respecto to input,
                shape(state_size, input_size)
            l_x (numpy.ndarray): gradient of cost with respecto to state,
                shape(state_size, )
            l_u (numpy.ndarray): gradient of cost with respecto to input,
                shape(input_size, )
            l_xx (numpy.ndarray): hessian of cost with respecto to state,
                shape(state_size, state_size)
            l_uu (numpy.ndarray): hessian of cost with respecto to input,
                shape(input_size, input_size)
            l_ux (numpy.ndarray): hessian of cost with respect
                to state and input, shape(input_size, state_size)
            V_x (numpy.ndarray): gradient of Value function,
                shape(state_size, )
            V_xx (numpy.ndarray): hessian of Value function,
                shape(state_size, state_size)
        Returns:
            Q_x (numpy.ndarray): gradient of Q function, shape(state_size, )
            Q_u (numpy.ndarray): gradient of Q function, shape(input_size, )
            Q_xx (numpy.ndarray): hessian of Q fucntion,
                shape(state_size, state_size)
            Q_ux (numpy.ndarray): hessian of Q fucntion,
                shape(input_size, state_size)
            Q_uu (numpy.ndarray): hessian of Q fucntion,
                shape(input_size, input_size)
        """
        # get size
        state_size = len(l_x)

        Q_x = l_x + np.dot(f_x.T, V_x)
        Q_u = l_u + np.dot(f_u.T, V_x)
        Q_xx = l_xx + np.dot(np.dot(f_x.T, V_xx), f_x)

        reg = self.mu * np.eye(state_size)
        Q_ux = l_ux + np.dot(np.dot(f_u.T, (V_xx + reg)), f_x)
        Q_uu = l_uu + np.dot(np.dot(f_u.T, (V_xx + reg)), f_u)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
