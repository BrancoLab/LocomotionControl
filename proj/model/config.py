import numpy as np

class Config:
    # ----------------------------- Simulation params ---------------------------- #

    dt  = 0.01
    
    # -------------------------------- Cost params ------------------------------- #
    STATE_SIZE = 5
    INPUT_SIZE = 2

    R = np.diag([0.1, 0.1]) # control cost
    Q = np.diag([2.5, 2.5, 2.5, 2.5, 0]) # state cost
    Sf = np.diag([2.5, 2.5, 2.5, 2.5, 0]) # final state cost

    # ------------------------------- Mouse params ------------------------------- #

    mouse = dict(
        L = 5, # half body width | cm
        R = 1, # radius of wheels | cm
        d = 0.2, # distance between axel and CoM | cm
        length = 6, # cm
        m = round(24/9.81, 2), # mass | g
        m_w = round(2/9.81, 2), # mass of wheels/legs |g
    )

    # ------------------------------ Goal trajectory ----------------------------- #

    trajectory = dict( # parameters of the goals trajectory
        nsteps = 200, 
        distance = 20,
        max_speed = 20,
        min_speed = 2,
    )

    # ------------------------------ Planning params ----------------------------- #    
    planning = dict( # params used to compute goal states to be used for control
        prediction_length = 20,
        n_ahead = 5, # start prediction states from N steps ahead
    )

    # ------------------------------ Control params ------------------------------ #
    iLQR = dict(
            max_iter = 500, 
            init_mu = 1.,
            mu_min = 1e-6,
            mu_max = 1e10,
            init_delta = 2.,
            threshold = 1e-6,
    )