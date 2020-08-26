from natu.units import g, m, s
from collections import namedtuple
import numpy as np

class Config():
    
    # Simulation and plotting params
    params = dict(
        duration = 30, # s
        dt = .25, # s | DT of goal trace, not simulation.
        distance = 100, # cm 
        max_speed = 70, # cm/s
        min_speed = 40, # cm /s 
    )

    mouse = dict(
        width = 3, # cm
        length = 6, # cm
        weight = 24, # g
        color=[.2, .2, .2],
    )

    alpha = 1 # numeric value to ensure dividend is != 0

    # General parameters
    ENV_NAME = "allocentric_2d"
    TYPE = "Nonlinear"
    N_AHEAD = 3
    TASK_HORIZON = 500
    PRED_LEN = 20
    
    # Model parameters
    m = 3 # grams
    STATE_SIZE = 4
    INPUT_SIZE = 2
    DT = 0.01

    # cost parameters
    R = np.diag([0.1, 0.1])
    Q = np.diag([2.5, 2.5, 2, 1])
    Sf = np.diag([2.5, 2.5, .1, .1])
    
    # bounds
    INPUT_LOWER_BOUND = np.array([-100, -100])
    INPUT_UPPER_BOUND = np.array([300, 300])

    # useful vars
    _state = namedtuple('state', 'x, y, theta, v')
    _control = namedtuple('cost', 'L, R')

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