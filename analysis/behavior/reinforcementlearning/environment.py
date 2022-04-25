import sys

sys.path.append("./")

import gym
from gym import spaces
import numpy as np
from dataclasses import dataclass
from collections import namedtuple

from stable_baselines3.common.env_checker import check_env


from bike import Bicycle
from utils import inbounds
from track import Track

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

    metadata = {"render.modes": ["human"]}

    def __init__(self, S: float = 20, ds: float = 1, dt: float = 0.001):
        super(MTMEnv, self).__init__()

        # load track from json file
        self.track = Track()

        # get a bike object
        self.bike = Bicycle(self.track, *self.initial_conditions(), dt=dt)

        # Define action and observation space
        bounds = Boundaries()
        self.boundaries = bounds
        self.action_space = spaces.Box(
            low=[bounds["deltadot"].low, bounds["Fu"].low],
            high=[bounds["deltadot"].high, bounds["Fu"].high],
            shape=(2, 1),
            dtype=np.float64,
        )

        # Define observation space
        assert S % ds == 0, "S must be divisible by ds"
        N = int(S / ds)
        _low = [
            bounds[var].low for var in ("n", "psi", "delta", "u", "omega")
        ] + [-np.inf] * N
        _high = [
            bounds[var].high for var in ("n", "psi", "delta", "u", "omega")
        ] + [np.inf] * N
        self.observation_space = spaces.Box(
            low=_low, high=_high, shape=len(_high), dtype=np.float64
        )

    def initial_conditions(self):
        """
            Get initial conditions from track start
        """
        raise NotImplementedError

    def step(self, action):
        # check action bounds
        deltadot, Fu = action
        deltadot = inbounds(
            deltadot,
            self.boundaries["deltadot"].low,
            self.boundaries["deltadot"].high,
        )
        Fu = inbounds(
            Fu, self.boundaries["Fu"].low, self.boundaries["Fu"].high
        )

        # update bike
        self.bike.step(deltadot, Fu)
        self.bike.enforce_boundaries(self.boundaries())
        # bikestate = self.bike.state()

        # get curvature observation

        # return observation, reward, done, info

    # def reset(self):
    #     ...
    #     return observation  # reward, done, info can't be included

    # def render(self, mode='human'):
    #     ...

    # def close (self):
    #     ...


if __name__ == "__main__":
    env = MTMEnv()
    check_env(env)
