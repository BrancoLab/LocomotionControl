
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

