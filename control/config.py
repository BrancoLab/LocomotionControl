import numpy as np

dt = 0.005
px_to_cm = 1  # 1 / 8
PARAMS = dict(dt=dt, px_to_cm=px_to_cm)

MANAGER_CONFIG = dict(exp_name="TRACKING", live_plot=False,)

TRAJECTORY_CONFIG = dict(
    traj_type="tracking",  # tracking or simulated # ! CHECK BEFORE REAL
    n_steps=1000,
    min_dist=1,  # when within this distance from end, stop
)


MOUSE = dict(
    L=0.02,  # half body width | m
    R=0.005,  # radius of wheels | m
    d=0.02,  # distance between axel and CoM | m
    length=5 * px_to_cm,  # cm | just for plotting
    m=round(0.023 / 9.81, 2),  # mass | kg
    m_w=round(0.0078 / 9.81, 2),  # mass of wheels/legs | kg
)

CONTROL_CONFIG = dict(
    STATE_SIZE=7,
    controls_size=3,
    ANGLE_IDX=2,  # state vector index which is angle, used to fit diff in
    # control magnitude
    R_start=np.diag([1, 1, 1]) * 1,
    R_run=np.diag([1, 1, 1]) * 1,
    # positive controls
    W=np.diag([-1, -1, -1]) * 0,  # should be < 0
    # control smoothness
    Z_start=np.diag([1, 1, 1]) * 1e2,
    Z_run=np.diag([1, 1, 1]) * 1e2,
    # state error cost
    # state cost | x, y, theta, v, omega, taul, taur
    Q=np.diag([500, 500, 200, 100, 1500, 0, 0]) * 1e-6,
)

# params used to compute goal states to be used for control
PLANNING_CONFIG = dict(
    prediction_length_start=60,  # prediction length for the first few steps
    prediction_length_run=60,  # length after that
    n_ahead=15,  # start prediction states from N steps ahead
)

iLQR_CONFIG = dict(
    max_iter=1000,  # number of iterations for descent
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
    {
        k: v if isinstance(v, int) else str(np.diag(v))
        for k, v in CONTROL_CONFIG.items()
    },
    PARAMS,
)
