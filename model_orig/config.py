import numpy as np

class AlloConfig():
    # parameters
    ENV_NAME = "allocentric_2d"

    TYPE = "Nonlinear"
    N_AHEAD = 1
    TASK_HORIZON = 500
    PRED_LEN = 20
    
    STATE_SIZE = 4
    INPUT_SIZE = 2
    DT = 0.01

    # cost parameters
    R = np.diag([0.01, 0.01])
    Q = np.diag([2.5, 2.5, 0.01, 0.01])
    Sf = np.diag([2.5, 2.5, 0.01, 0.01])
    
    # bounds
    INPUT_LOWER_BOUND = np.array([-1.5, -3.14])
    INPUT_UPPER_BOUND = np.array([1.5, 3.14])

    m = 10

    def __init__(self):
        """ 
        """
        # opt configs
        self.opt_config = {
            "Random": {
                "popsize": 5000
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 15,
                "alpha": 0.3,
                "init_var":1.,
                "threshold":0.001
            },
            "MPPI":{
                "beta" : 0.6,
                "popsize": 5000,
                "kappa": 0.9,
                "noise_sigma": 0.5,
            },
            "MPPIWilliams":{
                "popsize": 5000,
                "lambda": 1,
                "noise_sigma": 1.,
            },
           "iLQR":{
                "max_iter": 500,
                "init_mu": 1.,
                "mu_min": 1e-6,
                "mu_max": 1e10,
                "init_delta": 2.,
                "threshold": 1e-6,
           },
           "DDP":{
                "max_iter": 500,
                "init_mu": 1.,
                "mu_min": 1e-6,
                "mu_max": 1e10,
                "init_delta": 2.,
                "threshold": 1e-6,
           },
           "NMPC-CGMRES":{
           },
           "NMPC-Newton":{
           },
        } 


    @staticmethod
    def input_cost_fn(u):
        """ input cost functions
        Args:
            u (numpy.ndarray): input, shape(pred_len, input_size)
                or shape(pop_size, pred_len, input_size)
        Returns:
            cost (numpy.ndarray): cost of input, shape(pred_len, input_size) or
                shape(pop_size, pred_len, input_size)
        """
        return (u**2) * np.diag(AlloConfig.R)

    @staticmethod
    def state_cost_fn(x, g_x):
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
        return ((diff)**2) * np.diag(AlloConfig.Q)

    @staticmethod
    def terminal_state_cost_fn(terminal_x, terminal_g_x):
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
        
        return ((terminal_diff)**2) * np.diag(AlloConfig.Sf)