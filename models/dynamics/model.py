import numpy as np
# from loky import get_reusable_executor
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 

from control.models.model import Model
from control.common.utils import fit_angle_in_range

from control.helper import timeit

class Model(Model):
    
    last_dxdt = None
    nu = None

    def __init__(self, config, symbolic):
        """
        """
        super(Model, self).__init__()
        self.symbolic = symbolic
        self.mouse = config.mouse

        self.dt = config.DT
        self.Q = config.Q
        self.R = config.R
        self.Sf = config.Sf

        self._state = config._state
        self._control = config._control
        self.alpha = config.alpha

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
                                curr_x[:, 2],)
            u = self._control(u[:, 0], u[:, 1])
        else:
            u = self._control(*u)
            x = self._state(*curr_x)

        if self.last_dxdt is None:
            self.last_dxdt = self._state(*np.zeros_like(curr_x))
        if self.nu is None:
            self.nu = self._control(*np.zeros(2))

        values = dict(
            theta = x.theta,
            thetadot = self.last_dxdt.theta,
            tau_r = u.R,
            tau_l = u.L, 
            eta_r = self.nu.R,
            eta_l = self.nu.L
        )
        values = {**values, **self.mouse}

        # compute dxdt
        dxdt = self.symbolic.compute_xdot(
                    values['R'],
                    values['L'],
                    values['m'],
                    values['d'],
                    values['eta_r'],
                    values['eta_l'],
                    values['theta'],
                    values['thetadot'],
                    values['tau_r'],
                    values['tau_l'])
        dxdt = dxdt.ravel()

        # update nu and last dx
        if len(curr_x.shape) == 1:
            self.last_dxdt = self._state(*dxdt)
            
            dnudt = self.symbolic.eval(self.symbolic.dnudt, values)
            self.nu = self._control(*(np.array(dnudt).astype(np.float32).ravel() * self.dt + self.nu))

        if np.any(np.isnan(dxdt)):
            print(f'Current: {[round(p,2) for p in curr_x]}')
            print(f'dxdt: {[round(p,2) for p in dxdt]}')
            print(f'control: {[round(p,2) for p in u]}')
            raise ValueError('nan in dxdt, what the heck')

        # next state
        next_x = dxdt * self.dt + curr_x
        return next_x


# ------------------------------ MODEL GRADIENTS ----------------------------- #
    def calc_grad(self, xs, us, dt, func, dertype='x'):
        """
            Calculates the gradient with respect to either
            x or u based on which `func` from self.symbolic
            is passed

        """

        # prep values
        theta = xs[:, 2]
        thetadot = np.ones_like(theta) * self.last_dxdt.theta
        tau_r = us[:, 0]
        tau_l = us[:, 1]
        eta_r = np.ones_like(theta) * self.nu[0]
        eta_l = np.ones_like(theta) * self.nu[1]

        x = np.zeros_like(theta)
        y = np.zeros_like(theta)
        R = np.ones_like(theta) * self.mouse['R']
        L = np.ones_like(theta) * self.mouse['L']
        m = np.ones_like(theta) * self.mouse['m']
        d = np.ones_like(theta) * self.mouse['d']

        # reshape
        (_, state_size) = xs.shape
        (pred_len, input_size) = us.shape

        if dertype == 'x':
            shape = (pred_len, state_size, state_size)
            f =  self.symbolic.compute_xdot_dx(shape, R, L, m, d, eta_r, eta_l, theta, thetadot, tau_r, tau_l)
        else:
            shape = (pred_len, state_size, input_size)
            f = self.symbolic.compute_xdot_du(shape, R, L, m, d, eta_r, eta_l, theta, thetadot, tau_r, tau_l)

        # Check for nan
        if np.any(np.isnan(f)):
            raise ValueError("Found nans in the derivative of ", dertype)

        # return
        if dertype == 'x':
            return f * dt + np.eye(state_size)  # to discrete form
        else:
            return f * dt

    def calc_f_x(self, xs, us, dt):
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
        return self.calc_grad(xs, us, dt, self.symbolic.vec_xdot_dx, dertype='x')

    def calc_f_u(self, xs, us, dt):
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
        return self.calc_grad(xs, us, dt, self.symbolic.vec_xdot_du, dertype='u')


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


