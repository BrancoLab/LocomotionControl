import numpy as np

from PythonLinearNonlinearControl.models.model import Model

class AlloModel(Model):
    """ two wheeled model
    """
    def __init__(self, config):
        """
        """
        super(AlloModel, self).__init__()
        self.m = config.m
        self.dt = config.DT

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
        if len(u.shape) == 1:
            raise NotImplementedError
            # B = np.array([[np.cos(curr_x[-1]), 0.],
            #               [np.sin(curr_x[-1]), 0.],
            #               [0., 1.]])
            # # calc dot
            # x_dot = np.matmul(B, u[:, np.newaxis])
            # # next state
            # next_x = x_dot.flatten() * self.dt + curr_x

            # return next_x

        elif len(u.shape) == 2:
            (pop_size, state_size) = curr_x.shape
            (_, input_size) = u.shape

            x_dot = curr_x[:, -1] * np.cos(curr_x[:, 2]) # v * cos(theta)
            y_dot = curr_x[:, -1] * np.sin(curr_x[:, 2]) # v * sin(theta)
            theta_dot = (u[:, 0] - u[:, 1])/self.m       # (U.L - U.R)/m
            v_dot = (u[:, 0] + u[:, 1])/self.m * (1 - (u[:, 0]-u[:, 1])/(u[:, 0]+u[:, 1]))

            dxdt = np.zeros_like(curr_x)
            dxdt[:, 0] = x_dot
            dxdt[:, 1] = y_dot
            dxdt[:, 2] = theta_dot
            dxdt[:, 3] = v_dot

            # next state
            next_x = dxdt * self.dt + curr_x

            return next_x
    
    # @staticmethod
    # def calc_f_x(xs, us, dt):
    #     """ gradient of model with respect to the state in batch form
    #     Args:
    #         xs (numpy.ndarray): state, shape(pred_len+1, state_size)
    #         us (numpy.ndarray): input, shape(pred_len, input_size,)
        
    #     Return:
    #         f_x (numpy.ndarray): gradient of model with respect to x,
    #             shape(pred_len, state_size, state_size)
    #     Notes:
    #         This should be discrete form !!
    #     """ 
    #     # get size
    #     (_, state_size) = xs.shape
    #     (pred_len, _) = us.shape

    #     f_x = np.zeros((pred_len, state_size, state_size))
    #     f_x[:, 0, 2] = -np.sin(xs[:, 2]) * us[:, 0]
    #     f_x[:, 1, 2] = np.cos(xs[:, 2]) * us[:, 0]

    #     return f_x * dt + np.eye(state_size)  # to discrete form

    # @staticmethod
    # def calc_f_u(xs, us, dt):
    #     """ gradient of model with respect to the input in batch form
    #     Args:
    #         xs (numpy.ndarray): state, shape(pred_len+1, state_size)
    #         us (numpy.ndarray): input, shape(pred_len, input_size,)
        
    #     Return:
    #         f_u (numpy.ndarray): gradient of model with respect to x,
    #             shape(pred_len, state_size, input_size)
    #     Notes:
    #         This should be discrete form !!
    #     """ 
    #     # get size
    #     (_, state_size) = xs.shape
    #     (pred_len, input_size) = us.shape

    #     f_u = np.zeros((pred_len, state_size, input_size))
    #     f_u[:, 0, 0] = np.cos(xs[:, 2])
    #     f_u[:, 1, 0] = np.sin(xs[:, 2])
    #     f_u[:, 2, 1] = 1.

    #     return f_u * dt  # to discrete form

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

    #     f_uu = np.zeros((pred_len, state_size, input_size, input_size))

    #     return f_uu * dt

