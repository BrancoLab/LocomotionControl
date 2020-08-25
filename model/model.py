import numpy as np

from control.models.model import Model

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
            print(f'Current: {[round(p,2) for p in curr_x]}')
            print(f'dxdt: {[round(p,2) for p in dxdt]}')
            print(f'control: {[round(p,2) for p in u]}')

        # next state
        next_x = dxdt * self.dt + curr_x
        return next_x

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
        diff = x - g_x
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
        terminal_diff = terminal_x  - terminal_g_x
        return ((terminal_diff)**2) * np.diag(self.Sf)