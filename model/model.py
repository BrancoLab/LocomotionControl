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
        (pop_size, state_size) = curr_x.shape
        (_, input_size) = u.shape

        x = self._state(curr_x[:, 0], 
                            curr_x[:, 1]
                            curr_x[:, 2], 
                            curr_x[:, 3])
        u = self._control(u[:, 0], u[:, 1])

        # x_dot = curr_x[:, -1] * np.cos(curr_x[:, 2]) # v * cos(theta)
        # y_dot = curr_x[:, -1] * np.sin(curr_x[:, 2]) # v * sin(theta)
        # theta_dot = (u[:, 0] - u[:, 1])/self.m       # (U.L - U.R)/m
        # v_dot = (u[:, 0] + u[:, 1])/self.m * (1 - (u[:, 0]-u[:, 1])/(u[:, 0]+u[:, 1]))

        dxdt = np.zeros_like(curr_x)
        dxdt[:, 0] = x.v * np.cos(x.theta)
        dxdt[:, 1] = x.v * np.sin(x.theta)
        dxdt[:, 2] = (u.R - u.L)/self.m
        dxdt[:, 3] = (u.R + u.L)/self.m * (1 - (u.R - u.L)/(u.R+u.L))

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