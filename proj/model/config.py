import numpy as np

# ------------------------------- Control params ------------------------------ #
control_params = dict(
    STATE_SIZE=5,
    INPUT_SIZE=2,
    ANGLE_IDX=2,  # state vector index which is angle, used to fit diff in
    R=np.diag([1.0e-7, 1.0e-7]),  # control cost
    Q=np.diag([30, 30, 30, 10, 0]),  # state cost | x, y, theta, v, omega
    Sf=np.diag([0, 0, 0, 0, 0]),  # final state cost
)


# ---------------------------------------------------------------------------- #
#                                    CONFIG                                    #
# ---------------------------------------------------------------------------- #


class Config:
    # ----------------------------- Simulation params ---------------------------- #
    SIMULATION_NAME = "cleaned"

    USE_FAST = True  # if true use cumba's methods
    SPAWN_TYPE = "trajectory"
    LIVE_PLOT = False

    dt = 0.005

    # ------------------------------ Goal trajectory ----------------------------- #

    trajectory = dict(  # parameters of the goals trajectory
        name="tracking",
        # ? For artificial trajectories
        n_steps=500,
        distance=150,
        max_speed=100,
        min_speed=80,
        min_dist=0,  # if agent is within this distance from trajectory end the goal is considered achieved
        # ? for trajectories from data
        px_to_cm=1 / 8,  # convert px values to cm
        resample=True,  # if True when using tracking trajectory resamples it
        max_deg_interpol=8,  # if using track fit a N degree polynomial to daa to smoothen
        randomize=False,  # if true when using tracking it pulls a random trial
        trial_n=0,  # if not picking a random trial, which one should we use
        dt=0.005,  # used to simulate trajectories, should match simulation
    )

    # ------------------------------ Planning params ----------------------------- #
    planning = dict(  # params used to compute goal states to be used for control
        prediction_length=50,
        n_ahead=5,  # start prediction states from N steps ahead
    )

    # --------------------------------- Plotting --------------------------------- #
    traj_plot_every = 5

    # ------------------------------ LQR     params ------------------------------ #
    iLQR = dict(
        max_iter=500,
        init_mu=1.0,
        mu_min=1e-6,
        mu_max=1e10,
        init_delta=2.0,
        threshold=1e-6,
    )

    def __init__(self,):
        if self.trajectory["dt"] != self.dt:
            raise ValueError(
                "Trajectory dt and simulation dt dont match, forgot something Fede?"
            )

        # mouse params
        self.mouse = dict(
            L=2,  # half body width | cm
            R=1.5,  # radius of wheels | cm
            d=2,  # distance between axel and CoM | cm
            length=2,  # cm
            m=round(23 / 9.81, 2),  # mass | g
            m_w=round(7.8 / 9.81, 2),  # mass of wheels/legs |g
        )
        # Adjust mouse length for plotting
        self.mouse["length"] = (
            self.mouse["length"] * self.trajectory["px_to_cm"]
        )

        # control
        self.STATE_SIZE = control_params["STATE_SIZE"]
        self.INPUT_SIZE = control_params["INPUT_SIZE"]
        self.ANGLE_IDX = control_params["ANGLE_IDX"]
        self.R = control_params["R"]
        self.Q = control_params["Q"]
        self.Sf = control_params["Sf"]

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
