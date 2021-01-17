import numpy as np

dt = 0.005
px_to_cm = 1 / 8


MANAGER_CONFIG = dict(
    exp_name="planning_ahead_straight_really", live_plot=False, use_fast=True,
)

TRAJECTORY_CONFIG = dict(
    traj_type="simulated",  # tracking or simulated
    n_steps=1000,
    min_dist=5,  # when within this distance from end, stop
)


MOUSE = dict(
    L=2,  # half body width | cm
    R=1.5,  # radius of wheels | cm
    d=2,  # distance between axel and CoM | cm
    length=2 * px_to_cm,  # cm
    m=round(23 / 9.81, 2),  # mass | g
    m_w=round(7.8 / 9.81, 2),  # mass of wheels/legs |g
)

CONTROL_CONFIG = dict(
    STATE_SIZE=5,
    INPUT_SIZE=2,
    ANGLE_IDX=2,  # state vector index which is angle, used to fit diff in
    R=np.diag([1.0e-6, 1.0e-6]),  # control cost
    Q=np.diag([30, 30, 30, 10, 0]),  # state cost | x, y, theta, v, omega
    Sf=np.diag([0, 0, 0, 0, 0]),  # final state cost
)
CONTROL_CONFIG_FLAT = dict(  # for easy saving
    STATE_SIZE=5,
    INPUT_SIZE=2,
    ANGLE_IDX=2,  # state vector index which is angle, used to fit diff in
    R=[1.0e-6, 1.0e-6],  # control cost
    Q=[30, 30, 30, 10, 0],  # state cost | x, y, theta, v, omega
    Sf=[0, 0, 0, 0, 0],  # final state cost
)

PLANNING_CONFIG = dict(  # params used to compute goal states to be used for control
    prediction_length=100,
    n_ahead=25,  # start prediction states from N steps ahead
)

iLQR_CONFIG = dict(
    max_iter=500,
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
    CONTROL_CONFIG_FLAT,
    PLANNING_CONFIG,
    iLQR_CONFIG,
    dict(dt=dt, px_to_cm=px_to_cm),
)
