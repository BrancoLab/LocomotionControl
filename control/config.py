import numpy as np

dt = 0.005
px_to_cm = 1 / 8


MANAGER_CONFIG = dict(exp_name="new_model", live_plot=True,)

TRAJECTORY_CONFIG = dict(
    traj_type="simulated",  # tracking or simulated # ! CHECK BEFORE REAL
    n_steps=2000,
    min_dist=5,  # when within this distance from end, stop
)


MOUSE = dict(
    L=2,  # half body width | cm
    R=1.5,  # radius of wheels | cm
    d=2,  # distance between axel and CoM | cm
    length=10 * px_to_cm,  # cm | just for plotting
    m=round(23 / 9.81, 2),  # mass | g
    m_w=round(7.8 / 9.81, 2),  # mass of wheels/legs |g
)

CONTROL_CONFIG = dict(
    STATE_SIZE=7,
    INPUT_SIZE=3,
    ANGLE_IDX=2,  # state vector index which is angle, used to fit diff in
    R=np.diag([1, 1, 1]),  # control cost
    W=np.diag(
        [-20, -20, -20]
    ),  # control negative cost | should be < 0 | penalizes negative controls
    Q=np.diag(
        [30, 30, 30, 30, 0, 0, 0]
    ),  # state cost | x, y, theta, v, omega, taul, taur
)

PLANNING_CONFIG = dict(  # params used to compute goal states to be used for control
    prediction_length=50,
    n_ahead=50,  # start prediction states from N steps ahead
)

iLQR_CONFIG = dict(
    max_iter=1000,
    init_mu=1.0,
    mu_min=1e-6,
    mu_max=1e10,
    init_delta=2.0,
    threshold=1e-6,
)

all_configs = (
    MANAGER_CONFIG,
    TRAJECTORY_CONFIG,
    MOUSE,
    PLANNING_CONFIG,
    iLQR_CONFIG,
    dict(dt=dt, px_to_cm=px_to_cm),
)
