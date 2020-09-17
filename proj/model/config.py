import numpy as np


class Config:
    USE_FAST = True  # if true use cumba's methods
    SPAWN_TYPE = "trajectory"

    # ----------------------------- Simulation params ---------------------------- #
    dt = 0.01

    # -------------------------------- Cost params ------------------------------- #
    STATE_SIZE = 5
    INPUT_SIZE = 2

    R = np.diag([1.5, 1.5])  # control cost
    Q = np.diag([5, 5, 1, 4.5, 0])  # state cost | x, y, theta, v, omega
    Sf = np.diag([0, 0, 0, 0, 0])  # final state cost

    # STATE_SIZE = 4
    # INPUT_SIZE = 2

    # R = np.diag([0.05, 0.05])  # control cost
    # Q = np.diag([2.5, 2.5, 0, 0])  # state cost | r, omega, v, omega
    # Sf = np.diag([2.5, 2.5, 0, 0])  # final state cost

    # ------------------------------- Mouse params ------------------------------- #

    mouse = dict(
        L=1.5,  # half body width | cm
        R=1,  # radius of wheels | cm
        d=0.1,  # distance between axel and CoM | cm
        length=3,  # cm
        m=round(20 / 9.81, 2),  # mass | g
        m_w=round(2 / 9.81, 2),  # mass of wheels/legs |g
    )

    # ------------------------------ Goal trajectory ----------------------------- #

    trajectory = dict(  # parameters of the goals trajectory
        name="tracking",
        nsteps=1000,
        distance=150,
        max_speed=100,
        min_speed=80,
        min_dist=20,  # if agent is within this distance from trajectory end the goal is considered achieved
        skip=0,
        resample=True,  # if True when using tracking trajectory resamples it
        max_deg_interpol=8,  # if using track fit a N degree polynomial to daa to smoothen
        randomize=True,  # if true when using tracking it pulls a random trial
    )

    # ------------------------------ Planning params ----------------------------- #
    planning = dict(  # params used to compute goal states to be used for control
        prediction_length=80,
        n_ahead=5,  # start prediction states from N steps ahead
    )

    # --------------------------------- Plotting --------------------------------- #
    traj_plot_every = 15

    # ------------------------------ Control params ------------------------------ #
    iLQR = dict(
        max_iter=500,
        init_mu=1.0,
        mu_min=1e-6,
        mu_max=1e10,
        init_delta=2.0,
        threshold=1e-6,
    )

    def config_dict(self):
        return dict(
            dt=self.dt,
            STATE_SIZE=self.STATE_SIZE,
            INPUT_SIZE=self.INPUT_SIZE,
            R=list(np.diag(self.R).tolist()),
            Q=list(np.diag(self.Q).tolist()),
            Sf=list(np.diag(self.Sf).tolist()),
            mouse=self.mouse,
            trajectory=self.trajectory,
            planning=self.planning,
            iLQR=self.iLQR,
        )
