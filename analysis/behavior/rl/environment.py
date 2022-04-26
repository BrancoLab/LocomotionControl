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


from bike import Bicycle
from utils import inbounds, unnormalize
from track import Track

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
    delta: boundary = boundary(low=r(-80), high=r(80))
    u: boundary = boundary(low=10, high=80)
    v: boundary = boundary(low=-10, high=10)
    omega: boundary = boundary(low=r(-400), high=r(400))

    deltadot: boundary = boundary(low=-8, high=8)
    Fu: boundary = boundary(low=-3000, high=4500)

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

    def __init__(self, horizon: float = 20, ds: float = 1, dt: float = 0.001, log_dir=None):
        super(MTMEnv, self).__init__()

        if log_dir is not None:
            recorder.start(base_folder=log_dir)

        # load track from json file
        self.track = Track()

        # get a bike object
        self.dt = dt
        self.bike = Bicycle(self.track, *self.initial_conditions(), dt=dt)

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
        _low = [
            bounds[var].low for var in self._obs_keys
        ] + [-np.inf] * (N+1)
        _high = [
            bounds[var].high for var in self._obs_keys
        ] + [np.inf] * (N+1)

        self.observation_space = spaces.Box(
            low=np.array(_low), high=np.array(_high), shape=(len(_high),), dtype=np.float64
        )

        self.n_curv_obs = N
        self.horizon = horizon
        logger.debug("Environment initialized")

    def initial_conditions(self):
        # returns the bike's state at the beginning of the track
        state = [
            self.track.x[0],
            self.track.y[0],
            10.0, # u
            0.0, # delta
            0.0, # v
            self.track.theta[0],
            0.0, # omega
        ]
        return state

    def unnormalize_action(self, action):
        """
            convert action from [-1, 1] to real range
        """
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
        obs["s"] = self.track.s(obs["x"], obs["y"])
        del obs["x"]; del obs["y"]; del obs["theta"]
        
        # get curvature observations
        svals = np.linspace(obs["s"], obs["s"]+self.horizon, self.n_curv_obs+1)[0:-1]
        for i, s in enumerate(svals):
            obs[f"curv_{i}"] = self.track.curvature(s)

        return np.hstack(list(obs.values())).astype(np.float64)

    def step(self, action):
        
        # check action bounds
        deltadot, Fu = self.unnormalize_action(action)

        # update bike
        self.bike.step(deltadot, Fu)
        self.bike.enforce_boundaries(self.boundaries)

        # get observations
        observation = self.get_observation()

        # get reward
        bike_s = self.bike.s()
        reward = bike_s - self._bike_prev_s
        self._bike_prev_s = bike_s

        # get other info
        done = bool(bike_s > 255)
        info = {}
        # logger.debug(f"Env step, action: {action}, reward: {reward}, done: {done}")
        return observation, reward, done, info


    def reset(self) -> np.ndarray:
        self.render_init()
        self._bike_prev_s = 0.0

        # re initialize bike state
        self.bike.reset(*self.initial_conditions())
        logger.debug("Environment reset")
        return self.get_observation()

    def render_init(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 12))
        self.ax.axis("equal")
        self.ax.axis("off")
        imgpath = "analysis/behavior/src/Hairpin.png"

        # load and plot image
        self.img = plt.imread(imgpath)
        self.ax.imshow(self.img, extent=[0, 40, 0, 60])

    def render(self, mode='rgb_array'):
        # clear self.ax
        self.ax.clear()

        # plot image
        self.ax.imshow(self.img, extent=[0, 40, 0, 6], origin="lower")

        # plot bike
        x, y, theta = self.bike.x, self.bike.y, self.bike.theta
        self.ax.plot(x, y, "o", color="red")
        self.ax.plot([x, x+0.5*np.cos(theta)], [y, y+0.5*np.sin(theta)], color="black")
        
        # return as rgb array
        return self.fig.canvas.renderer.buffer_rgba()


if __name__ == "__main__":
    env = MTMEnv()
    check_env(env)
