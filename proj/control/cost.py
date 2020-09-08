import numpy as np
from proj.control.utils import fit_angle_in_range


class Cost:
    @staticmethod
    def fit_diff_in_range(diff_x):
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
            diff_x[:, :, -2] = fit_angle_in_range(diff_x[:, :, -2])
        elif len(diff_x.shape) == 2:
            diff_x[:, -2] = fit_angle_in_range(diff_x[:, -2])
        elif len(diff_x.shape) == 1:
            diff_x[-2] = fit_angle_in_range(diff_x[-2])

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
        return ((diff) ** 2) * np.diag(self.Q)

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
