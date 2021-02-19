import numpy as np

dt = 0.005
px_to_cm = 1  # 1 / 8
PARAMS = dict(dt=dt, px_to_cm=px_to_cm)

MANAGER_CONFIG = dict(exp_name="TRACKING", live_plot=False,)

TRAJECTORY_CONFIG = dict(
    traj_type="tracking",  # tracking or simulated # ! CHECK BEFORE REAL
    n_steps=1000,
    min_dist=5,  # when within this distance from end, stop
)


MOUSE = dict(
    L=2,  # half body width | cm
    R=0.5,  # radius of wheels | cm
    d=2,  # distance between axel and CoM | cm
    length=5 * px_to_cm,  # cm | just for plotting
    m=23 / 9.81,  # mass | g
    m_w=8 / 9.81,  # mass of wheels/legs | g
)

CONTROL_CONFIG = dict(
    STATE_SIZE=7,
    controls_size=3,
    ANGLE_IDX=2,  # state vector index which is angle, used to fit diff in
    # control magnitude
    R=np.diag([1, 0.5, 0.5]) * 1e-3,
    # controls sparsity
    alpha=8e3,
    # control smoothness
    Z=np.diag([1, 1, 1]) * 1e2,
    # control positive
    W=np.diag([-1, -1, -1]) * 7e3,
    # state error cost
    # state cost | x, y, theta, v, omega, taul, taur
    Q=np.diag([1000, 1000, 200, 150, 1500, 0, 0]) * 1e4,
)

# params used to compute goal states to be used for control
PLANNING_CONFIG = dict(
    prediction_length=60,  # length after that
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
        k: v if isinstance(v, (int, float)) else str(np.diag(v))
        for k, v in CONTROL_CONFIG.items()
    },
    PARAMS,
)
