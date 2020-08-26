import numpy as np

from control.models.model import Model
from control.common.utils import fit_angle_in_range

class AlloModel(Model):
    """ two wheeled model
    """
    def __init__(self, config):
        """
        """
        super(AlloModel, self).__init__()
        self.m = config.m
        self.dt = config.DT
        self.Q = config.Q
        self.R = config.R
        self.Sf = config.Sf
        self._state = config._state
        self._control = config._control

    def predict_next_state(self, curr_x, u):
        """ predict next state
        
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, ) or
                shape(pop_size, state_size)
            u (numpy.ndarray): input, shape(input_size, ) or
                shape(pop_size, input_size)
        Returns:
            next_x (numpy.ndarray): next state, shape(state_size, ) or
                shape(pop_size, state_size)
        """
        # Parse args
        if len(curr_x.shape) > 1:
            (pop_size, state_size) = curr_x.shape
            (_, input_size) = u.shape

            x = self._state(curr_x[:, 0], 
                                curr_x[:, 1],
                                curr_x[:, 2], 
                                curr_x[:, 3])
            u = self._control(u[:, 0], u[:, 1])
        else:
            u = self._control(*u)
            x = self._state(*curr_x)

        # Compute derivatives
        dx = x.v * np.cos(x.theta)
        dy = x.v * np.sin(x.theta)
        # dtheta = ((u.R - u.L)/self.m) * 500
        # dv = ((u.R + u.L)/self.m) * np.nan_to_num((1 - np.abs((u.R - u.L)/(u.R+u.L))))
        dtheta = u.L / self.m
        dv = u.R / self.m


        # Get dxdt 
        if len(curr_x.shape) > 1:
            dxdt = np.zeros_like(curr_x)
            dxdt[:, 0] = dx
            dxdt[:, 1] = dy
            dxdt[:, 2] = dtheta
            dxdt[:, 3] = dv
        else:
            dxdt = np.array([
                dx,
                dy,
                dtheta,
                dv
            ]).flatten()

            if np.any(np.isnan(dxdt)):
                raise ValueError('what the heck')
            # print(f'Current: {[round(p,2) for p in curr_x]}')
            # print(f'dxdt: {[round(p,2) for p in dxdt]}')
            # print(f'control: {[round(p,2) for p in u]}')

        # next state
        next_x = dxdt * self.dt + curr_x
        return next_x

    # --------------------------- STATE COST FUNCTIONS --------------------------- #

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
        return (u**2) * np.diag(self.R)

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
        return ((diff)**2) * np.diag(self.Q)

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
        terminal_diff = self.fit_diff_in_range(terminal_x  - terminal_g_x)
        return ((terminal_diff)**2) * np.diag(self.Sf)

  
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
            return 2. * (diff) * np.diag(self.Q)
        
        return (2. * (diff) * np.diag(self.Sf))[np.newaxis, :]

    # ---------------------------- COST FUN GRADIENTS ---------------------------- #

    def gradient_cost_fn_with_input(self, x, u):
        """ gradient of costs with respect to the input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        
        Returns:
            l_u (numpy.ndarray): gradient of cost, shape(pred_len, input_size)
        """
        return 2. * u * np.diag(self.R)

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
            return np.tile(2.*self.Q, (pred_len, 1, 1))               
        
        return np.tile(2.*self.Sf, (1, 1, 1))    

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

        return np.tile(2.*self.R, (pred_len, 1, 1))
    
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

# ------------------------------ MODEL GRADIENTS ----------------------------- #
    @staticmethod
    def calc_f_x(xs, us, dt):
        """ gradient of model with respect to the state in batch form
        Args:
            xs (numpy.ndarray): state, shape(pred_len+1, state_size)
            us (numpy.ndarray): input, shape(pred_len, input_size,)
        
        Return:
            f_x (numpy.ndarray): gradient of model with respect to x,
                shape(pred_len, state_size, state_size)
        Notes:
            This should be discrete form !!
        """ 
        # get size
        (_, state_size) = xs.shape
        (pred_len, _) = us.shape

        f_x = np.zeros((pred_len, state_size, state_size))
        f_x[:, 0, 2] = -np.sin(xs[:, 2]) * xs[:, 3]
        f_x[:, 0, 3] = np.cos(xs[:, 2])
        f_x[:, 1, 2] = np.cos(xs[:, 2]) * xs[:, 3]
        f_x[:, 1, 3] = np.sin(xs[:, 2])

        return f_x * dt + np.eye(state_size)  # to discrete form

    @staticmethod
    def calc_f_u(xs, us, dt):
        """ gradient of model with respect to the input in batch form
        Args:
            xs (numpy.ndarray): state, shape(pred_len+1, state_size)
            us (numpy.ndarray): input, shape(pred_len, input_size,)
        
        Return:
            f_u (numpy.ndarray): gradient of model with respect to x,
                shape(pred_len, state_size, input_size)
        Notes:
            This should be discrete form !!
        """ 
        # get size
        (_, state_size) = xs.shape
        (pred_len, input_size) = us.shape

        f_u = np.zeros((pred_len, state_size, input_size))

        f_u[:, 2, 0] = 1.
        f_u[:, 3, 1] = 1.

        return f_u * dt  # to discrete form

    # @staticmethod
    # def calc_f_xx(xs, us, dt):
    #     """ hessian of model with respect to the state in batch form
    #     Args:
    #         xs (numpy.ndarray): state, shape(pred_len+1, state_size)
    #         us (numpy.ndarray): input, shape(pred_len, input_size,)
        
    #     Return:
    #         f_xx (numpy.ndarray): gradient of model with respect to x,
    #             shape(pred_len, state_size, state_size, state_size)
    #     """
    #     # get size
    #     (_, state_size) = xs.shape
    #     (pred_len, _) = us.shape

    #     f_xx = np.zeros((pred_len, state_size, state_size, state_size))

    #     raise NotImplementedError
    #     f_xx[:, 0, 2, 2] = -np.cos(xs[:, 2]) * us[:, 0]
    #     f_xx[:, 1, 2, 2] = -np.sin(xs[:, 2]) * us[:, 0]

    #     return f_xx * dt

    # @staticmethod
    # def calc_f_ux(xs, us, dt):
    #     """ hessian of model with respect to state and input in batch form
    #     Args:
    #         xs (numpy.ndarray): state, shape(pred_len+1, state_size)
    #         us (numpy.ndarray): input, shape(pred_len, input_size,)
        
    #     Return:
    #         f_ux (numpy.ndarray): gradient of model with respect to x,
    #             shape(pred_len, state_size, input_size, state_size)
    #     """
    #     # get size
    #     (_, state_size) = xs.shape
    #     (pred_len, input_size) = us.shape

    #     f_ux = np.zeros((pred_len, state_size, input_size, state_size))

    #     raise NotImplementedError
    #     f_ux[:, 0, 0, 2] = -np.sin(xs[:, 2])
    #     f_ux[:, 1, 0, 2] = np.cos(xs[:, 2])

    #     return f_ux * dt
    
    # @staticmethod
    # def calc_f_uu(xs, us, dt):
    #     """ hessian of model with respect to input in batch form
    #     Args:
    #         xs (numpy.ndarray): state, shape(pred_len+1, state_size)
    #         us (numpy.ndarray): input, shape(pred_len, input_size,)
        
    #     Return:
    #         f_uu (numpy.ndarray): gradient of model with respect to x,
    #             shape(pred_len, state_size, input_size, input_size)
    #     """
    #     # get size
    #     (_, state_size) = xs.shape
    #     (pred_len, input_size) = us.shape

    #     raise NotImplementedError
    #     f_uu = np.zeros((pred_len, state_size, input_size, input_size))

    #     return f_uu * dt

