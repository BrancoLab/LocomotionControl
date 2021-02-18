import numpy as np

from sympy import MatrixSymbol
from einops import repeat

from .utils import fit_angle_in_range
from .config import CONTROL_CONFIG


def calc_cost(
    X_hat, U_hat, U_prev, X_g, state_cost_fn, input_cost_fn,
):
    """ calculate the cost 

    Args:
        X_hat (numpy.ndarray): predicted state trajectory, 
            shape(pop_size, pred_len+1, state_size)
        U_hat (numpy.ndarray): inputs U trajectory,
            shape(pop_size, pred_len+1, controls_size)
        U_prev (numpy.ndarray): input at previous step, shape(pred_len, controls_size)
                or shape(pop_size, pred_len, controls_size)
        X_g (numpy.ndarray): goal state trajectory,
            shape(pop_size, pred_len+1, state_size)
        state_cost_fn (function): state cost fucntion
        input_cost_fn (function): input cost fucntion

    Returns:
        cost (numpy.ndarray): cost of the input sample, shape(pop_size, )
    """
    # state cost
    state_pred_par_cost = state_cost_fn(X_hat[:, 1:-1, :], X_g[:, 1:-1, :])
    state_cost = np.sum(np.sum(state_pred_par_cost, axis=-1), axis=-1)

    # act cost
    act_pred_par_cost = input_cost_fn(U_hat, U_prev)
    act_cost = np.sum(np.sum(act_pred_par_cost, axis=-1), axis=-1)

    return state_cost + act_cost


class Cost:
    def __init__(self):
        """
            Handles cost computation and its derivatives
        """
        # self.make_equations()

    def make_equations(self):
        """
            Latex representation of cost and it's derivative
        """
        n_states = len(CONTROL_CONFIG["Q"])
        n_controls = len(CONTROL_CONFIG["W"])

        # first symbolic for nice printing of the cost eq
        X = MatrixSymbol("X", n_states, 1)
        X_g = MatrixSymbol("X_g", n_states, 1)
        U = MatrixSymbol("U", n_controls, 1)
        U_prev = MatrixSymbol("U_p", n_controls, 1)

        Q = MatrixSymbol("Q", n_states, n_states)
        R = MatrixSymbol("R", n_controls, n_controls)
        W = MatrixSymbol("W", n_controls, 1)
        Z = MatrixSymbol("Z", n_controls, n_controls)

        dX = X - X_g
        dU = U - U_prev

        raise NotImplementedError("Dont know how to show l1 norm in sympy")

        self.cost_function = (
            dX.T * Q * dX + U.T * R * U + U.T * W + dU.T * Z * dU
        )

    def calc_cost(self, X, U, U_prev, X_g):
        """ calculate the cost of input U

        Args:
            X (numpy.ndarray): shape(state_size),
                current robot position
            U (numpy.ndarray): shape(pop_size, opt_dim), 
                input U
            U_prev (numpy.ndarray): input at previous step, shape(pred_len, controls_size)
                or shape(pop_size, pred_len, controls_size)
            X_g (numpy.ndarray): shape(pred_len, state_size),
                goal states
        Returns:
            costs (numpy.ndarray): shape(pop_size, )
        """
        # get size
        pop_size = U.shape[0]
        X_g = np.tile(X_g, (pop_size, 1, 1))

        # calc cost, X_hat.shape = (pop_size, pred_len+1, state_size)
        X_hat = self.model.predict_trajectory(X, U)

        # get particle cost
        costs = calc_cost(
            X_hat, U, U_prev, X_g, self.state_cost_fn, self.input_cost_fn,
        )

        return costs

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

    def input_cost_fn(self, U, U_prev):
        """ input cost functions
        Args:
            U (numpy.ndarray): input, shape(pred_len, controls_size)
                or shape(pop_size, pred_len, controls_size)
            U_prev (numpy.ndarray): input at previous step, shape(pred_len, controls_size)
                or shape(pop_size, pred_len, controls_size)
        Returns:
            cost (numpy.ndarray): cost of input, shape(pred_len, controls_size) or
                shape(pop_size, pred_len, controls_size)
        """
        dU = U - U_prev
        Uneg = U.copy()
        Uneg[Uneg > 0] = 0

        magnitude = (U ** 2) * self.R_
        smoothness = (dU ** 2) * self.Z_
        if len(U.shape) == 3:
            if U.shape[0] > 1:
                raise NotImplementedError("Ops")
            sparsness = self.alpha * np.linalg.norm(U[0, :, :], ord=1, axis=1)
            sparsness = repeat(
                sparsness, "k -> i k n", i=U.shape[0], n=U.shape[2]
            )
        else:
            raise NotImplementedError("ops")
        positive = Uneg * self.W_
        return magnitude + smoothness + sparsness + positive

    def state_cost_fn(self, X, X_g):
        """ state cost function
        Args:
            X (numpy.ndarray): state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
            X_g (numpy.ndarray): goal state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, shape(pred_len, state_size) or
                shape(pop_size, pred_len, state_size)


            Cost of state, is given bY
            (X - X_g)T * Q * (X - x_g)
        """
        diff = self.fit_diff_in_range(X - X_g)

        return (diff ** 2) * self.Q_

    def gradient_cost_fn_with_state(self, X, X_g):
        """ gradient of costs with respect to the state

        Args:
            X (numpy.ndarray): state, shape(pred_len, state_size)
            X_g (numpy.ndarray): goal state, shape(pred_len, state_size)
        
        Returns:
            l_x (numpy.ndarray): gradient of cost, shape(pred_len, state_size)
                or shape(1, state_size)
        """
        diff = self.fit_diff_in_range(X - X_g)
        return 2.0 * (diff) * self.Q_

    def gradient_cost_fn_with_input(self, X, U, U_prev):
        """ gradient of costs with respect to the input

        Args:
            X (numpy.ndarray): state, shape(pred_len, state_size)
            U (numpy.ndarray): goal state, shape(pred_len, controls_size)
            U_prev (numpy.ndarray): input at previous step, shape(pred_len, controls_size)
                or shape(pop_size, pred_len, controls_size)
        
        Returns:
            l_u (numpy.ndarray): gradient of cost, shape(pred_len, controls_size)
        """
        dU = U - U_prev
        return 2.0 * U * self.R_ + 2 * dU * self.Z_ + self.alpha + self.W_

    def hessian_cost_fn_with_state(self, X, X_g):
        """ hessian costs with respect to the state

        Args:
            X (numpy.ndarray): state, shape(pred_len, state_size)
            X_g (numpy.ndarray): goal state, shape(pred_len, state_size)
        
        Returns:
            l_xx (numpy.ndarray): gradient of cost,
                shape(pred_len, state_size, state_size) or
                shape(1, state_size, state_size) or
        """
        (pred_len, _) = X.shape
        return np.tile(2.0 * self.Q, (pred_len, 1, 1))

    def hessian_cost_fn_with_input(self, X, U):
        """ hessian costs with respect to the input

        Args:
            X (numpy.ndarray): state, shape(pred_len, state_size)
            U (numpy.ndarray): goal state, shape(pred_len, controls_size)
        
        Returns:
            l_uu (numpy.ndarray): gradient of cost,
                shape(pred_len, controls_size, controls_size)
        """
        (pred_len, _) = U.shape

        return np.tile(2.0 * self.R + 2 * self.Z, (pred_len, 1, 1))

    def hessian_cost_fn_with_input_state(self, X, U):
        """ hessian costs with respect to the state and input

        Args:
            X (numpy.ndarray): state, shape(pred_len, state_size)
            U (numpy.ndarray): goal state, shape(pred_len, controls_size)
        
        Returns:
            l_ux (numpy.ndarray): gradient of cost ,
                shape(pred_len, controls_size, state_size)
        """
        (_, state_size) = X.shape
        (pred_len, controls_size) = U.shape

        return np.zeros((pred_len, controls_size, state_size))
