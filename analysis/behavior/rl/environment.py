import sys

sys.path.append("./")

import gym
from gym import spaces
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from tpd import recorder
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg
from rich.pretty import install
from rich import print

from bike import Bicycle
from utils import unnormalize
from track import Track

install()
logger.remove()
logger.add(sys.stdout, level="INFO")

"""
openai gym environment for the reinforcement learning task
based on the MTM problem.
"""

r = np.radians
boundary = namedtuple("bound", "low, high",)


@dataclass
class Boundaries:
    """
        Convenience class for defining boundaries for the state/observation space.
    """

    n: boundary = boundary(low=-3, high=3)
    psi: boundary = boundary(low=r(-30), high=r(30))

    delta: boundary = boundary(low=r(-60), high=r(60))
    u: boundary = boundary(low=10, high=80)
    v: boundary = boundary(low=-12, high=12)
    omega: boundary = boundary(low=r(-600), high=r(600))

    deltadot: boundary = boundary(low=-4, high=4)
    Fu: boundary = boundary(low=-4500, high=4500)

    def __getitem__(self, item):
        return getattr(self, item)


class MTMEnv(gym.Env):
    """
        RL environment for MTM problem.
        The agent has 2 continuous actions: δ̇ and Fu with bounds as per the MTM problem in Julia.

        Observations include the model's state:
            - ψ, n: angular and lateral errors
            - δ, u: steering angle and speed
            - ω: angular velocity
            - s: track progression
        and a vector representing track curvature for N cm ahead at ds space intervals
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    _obs_keys = ("psi", "u", "v", "omega", "delta", "n")

    def __init__(
        self,
        horizon: float = 20,
        ds: float = 1,
        dt: float = 0.001,
        log_dir=None,
        max_n_steps=2000,
    ):
        super(MTMEnv, self).__init__()

        self.MAX_N_STEPS = max_n_steps

        if log_dir is not None:
            recorder.start(base_folder=log_dir)

        # load track from json file
        self.track = Track()

        # Define action and observation space
        bounds = Boundaries()
        self.boundaries = bounds

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32,
        )

        # Define observation space (multiply by N+1 to account for 's')
        assert horizon % ds == 0, "S must be divisible by ds"
        N = int(horizon / ds)
        _low = np.array([-np.inf] * len(self._obs_keys) + [-np.inf] * (N + 1))
        _high = -_low

        self.observation_space = spaces.Box(
            low=np.array(_low),
            high=np.array(_high),
            shape=(len(_high),),
            dtype=np.float64,
        )

        self.n_curv_obs = N
        self.horizon = horizon
        self.render_init()

        # get a bike object
        self.dt = dt
        self.bike = Bicycle(
            self.track, self.boundaries, *self.initial_conditions(), dt=dt
        )

        logger.debug("Environment initialized")

    def initial_conditions(self):
        # returns the bike's state at the beginning of the track
        state = [
            20.0,  # u
            0.0,  # delta
            0.0,  # v
            0.0,  # omega
            0.0,  # s
        ]
        return state

    def unnormalize_action(self, action):
        """
            convert action from [-1, 1] to real range
        """
        if isinstance(action, tuple):
            action = action[0]

        if len(action.shape) == 2:
            action = action.ravel()

        deltadot, Fu = action
        deltadot = unnormalize(
            deltadot,
            self.boundaries["deltadot"].low,
            self.boundaries["deltadot"].high,
        )
        Fu = unnormalize(
            Fu, self.boundaries["Fu"].low, self.boundaries["Fu"].high
        )
        return deltadot, Fu

    def get_observation(self) -> np.ndarray:
        obs = self.bike.state()
        # obs["s"] = self.track.s(obs["x"], obs["y"])
        del obs["x"]
        del obs["y"]
        del obs["theta"]

        # get curvature observations
        svals = np.linspace(
            obs["s"], obs["s"] + self.horizon, self.n_curv_obs + 1
        )[0:-1]
        for i, s in enumerate(svals):
            obs[f"curv_{i}"] = self.track.curvature(s)

        return np.hstack(list(obs.values())).astype(np.float64)

    def should_terminate(self):
        bike_s = self.bike.s

        if bike_s > 250:
            # logger.info("done because at end")
            return True

        if self.n_steps > self.MAX_N_STEPS:
            # logger.info(f"done because max steps reached: {self.n_steps}")
            return True

        width = (self.track.w(bike_s) - self.bike.width) / 2
        if self.bike.n > width or self.bike.n < -width:
            # logger.info("done because out of track width")
            return True

        if (
            self.bike.psi > self.boundaries.psi.high
            or self.bike.psi < self.boundaries.psi.low
        ):
            # logger.info("done because out of psi range")
            return True

        return False

    def step(self, action):
        self.n_steps += 1

        # check action bounds
        deltadot, Fu = self.unnormalize_action(action)

        # update bike
        self.bike.step(deltadot, Fu)

        # get observations
        observation = self.get_observation()

        # get reward
        bike_s = self.bike.s
        reward = bike_s - self._bike_prev_s
        self._bike_prev_s = bike_s

        # get other info
        done = self.should_terminate()
        info = {}
        # logger.debug(f"Env step, action: {action}, reward: {reward}, done: {done}")
        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        # clear self.axes["A"]
        for ax in self.axes.values():
            ax.clear()
        self.axes["A"].imshow(self.img, extent=[0, 40, 0, 60], origin="upper")

        # reset bike
        self._bike_prev_s = 0.0
        self.n_steps = 0

        # re initialize bike state
        self.bike.reset(*self.initial_conditions())
        # logger.debug("Environment reset")
        return self.get_observation()

    def render_init(self):
        self.fig = plt.figure(figsize=(8, 12), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        self.fig.tight_layout()

        # self.axes = self.fig.subplot_mosaic(
        #     """
        #         AABBCCHH
        #         AADDEEII
        #         AAFFGGLL
        #     """
        # )
        self.axes = self.fig.subplot_mosaic(
            """
                AA
                AA
                AA
            """
        )
        self.axes["A"].axis("equal")
        self.axes["A"].axis("off")
        imgpath = "analysis/behavior/src/Hairpin.png"

        # load and plot image
        self.img = plt.imread(imgpath)
        self.axes["A"].imshow(self.img, extent=[0, 40, 0, 60])

    def render(self, mode="rgb_array"):
        # plot bike
        x, y, theta = self.bike.x, self.bike.y, self.bike.theta

        print({**self.bike.state(), **{"k": self.bike.k()}})

        self.axes["A"].plot(
            [x, x + 2 * np.cos(theta)],
            [y, y + 2 * np.sin(theta)],
            color="black",
        )
        self.axes["A"].plot(x, y, "o", color="red")
        self.axes["A"].set(xticks=[], yticks=[])
        self.axes["A"].axis("off")

        # # plot speeds
        # # self.axes["B"].scatter(self.n_steps, v, label="v")
        # self.axes["B"].scatter(self.n_steps, u, marker="x", label="u")

        # # plot s
        # self.axes["C"].scatter(self.n_steps, self.bike.s(), label="s")

        # # plot n
        # self.axes["D"].scatter(self.n_steps, self.bike.n, marker="x", label="n")

        # # plot theta & delta
        # self.axes["E"].scatter(self.n_steps, np.degrees(theta), label="theta")
        # self.axes["E"].scatter(self.n_steps, np.degrees(self.bike.delta), marker="x", label="delta")

        # # plot omega
        # self.axes["F"].scatter(self.n_steps, np.degrees(self.bike.omega), label="omega")

        # # plot psi
        # self.axes["G"].scatter(self.n_steps, np.degrees(self.bike.psi), label="psi")

        # # plot Fu
        # self.axes["H"].scatter(self.n_steps, self.bike.Fu, label="Fu")

        # # plot deltadot
        # self.axes["I"].scatter(self.n_steps, self.bike.deltadot, label="deltadot")

        # if self.n_steps ==0:
        #     for ax in "BCDEFGHI":
        #         self.axes[ax].legend()

        # return as rgb array
        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        # convert to a NumPy array
        return np.asarray(buf)


if __name__ == "__main__":
    env = MTMEnv()
    check_env(env)
