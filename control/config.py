import numpy as np

dt = 0.005
px_to_cm = 1 / 8


MANAGER_CONFIG = dict(exp_name="new_cost", live_plot=False,)

TRAJECTORY_CONFIG = dict(
    traj_type="simulated",  # tracking or simulated # ! CHECK BEFORE REAL
    n_steps=1000,
    min_dist=5,  # when within this distance from end, stop
)


MOUSE = dict(
    L=2,  # half body width | cm
    R=0.5,  # radius of wheels | cm
    d=2,  # distance between axel and CoM | cm
    length=5 * px_to_cm,  # cm | just for plotting
    m=round(23 / 9.81, 2),  # mass | g
    m_w=round(7.8 / 9.81, 2),  # mass of wheels/legs |g
)

CONTROL_CONFIG = dict(
    STATE_SIZE=7,
    controls_size=3,
    ANGLE_IDX=2,  # state vector index which is angle, used to fit diff in
    # control magnitude
    R_start=np.diag([1, 1, 1]) * 1e-1,
    R_run=np.diag([1, 1, 1]) * 1e-1,
    # positive controls
    W=np.diag([-1, -1, -1]) * 1e2,  # should be < 0
    # control smoothness
    Z_start=np.diag([1, 1, 1]) * 0,
    Z_run=np.diag([1, 1, 1]) * 4e-1,
    # state error cost
    # state cost | x, y, theta, v, omega, taul, taur

    Q=np.diag([30, 30, 30, 20, 30, 0, 0]) * 1e4,
)

# params used to compute goal states to be used for control
PLANNING_CONFIG = dict(
    prediction_length_start=15,  # prediction length for the first few steps
    prediction_length_start=20,  # prediction length for the first few steps
    prediction_length_run=20,  # length for a few iters after start ones
    prediction_length_long=20,  # length after that
    n_ahead=5,  # start prediction states from N steps ahead
)

iLQR_CONFIG = dict(
    max_iter=500,  # number of iterations for descent
    init_mu=1.0,
    mu_min=1e-6,
    mu_max=1e20,
    init_delta=2.0,
    threshold=1e-6,  # when improvement speed < this, stop optimization
)

all_configs = (
    MANAGER_CONFIG,
    TRAJECTORY_CONFIG,
    MOUSE,
    PLANNING_CONFIG,
    iLQR_CONFIG,
    {k: str(v) for k, v in CONTROL_CONFIG.items()},
    dict(dt=dt, px_to_cm=px_to_cm),
)
