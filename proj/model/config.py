class Config:
    dt  = 0.01

    mouse = dict(
        L = 5, # half body width | cm
        R = 1, # radius of wheels | cm
        d = 0.1, # distance between axel and CoM | cm
        length = 6, # cm
        m = round(24/9.81, 2), # mass | g
    )