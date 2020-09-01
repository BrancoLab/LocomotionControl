from collections import namedtuple
import numpy as np

class Config():
    
    # Simulation params and plotting
    params = dict(
        duration = 30, # s
        dt = .75, # s | DT of goal trace, not simulation.
        distance = 10, # cm 
        max_speed = 2, # cm/s
        min_speed = 1, # cm /s 

        mouse_color = [.2, .2, .2],
    )

    # Mouse params
    mouse = dict(
        L = 1.5, # half body width | cm
        R = 1, # radius of wheels | cm
        d = 0.75, # distance between axel and CoM | cm
        length = 6, # cm
        m = round(24/9.81, 2), # mass | g
    )



    alpha = 1 # numeric value to ensure dividend is != 0

    # General parameters
    ENV_NAME = "dynamics"
    TYPE = "Nonlinear"
    N_AHEAD = 3
    TASK_HORIZON = 500
    PRED_LEN = 20
    
    # Model parameters
    STATE_SIZE = 3
    INPUT_SIZE = 2
    DT = 0.01

    # cost parameters
    R = np.diag([0.1, 0.1])
    Q = np.diag([0, 0, 2.5, 0])
    Sf = np.diag([2.5, 2.5, 2.5, 2.5])
    
    # bounds
    INPUT_LOWER_BOUND = np.array([-.2, -.2])
    INPUT_UPPER_BOUND = np.array([.2, .2])

    # useful vars
    _state = namedtuple('state', 'x, y, theta, s')
    _control = namedtuple('control', 'L, R')

    # Fitting algorithms params
    opt_config = {
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
                "max_iter": 500, # was 500
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