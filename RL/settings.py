from collections import namedtuple

BUFFER_SIZE = 1 * int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 0.0001  # learning rate of the actor
LR_CRITIC = 0.001  # learning rate of the critic
WEIGHT_DECAY = 0.0  # L2 weight decays

N_EPISODES = 10000
MAX_EPISODE_LEN = 1000
MIN_GOAL_DISTANCE = 10

NOISE_SCALE = 0.002


# INPUTS SCALES
lim = namedtuple("lim", "min, max")
LIMITS = {
    "x": lim(-300, 300),
    "y": lim(-300, 300),
    "r": lim(-10, 10),
    "psy": lim(-20, 20),
    "v": lim(-300, 300),
    "o": lim(-360, 360),
    "dv": lim(-150, 150),
    "do": lim(-250, 250),
    "tau_l": lim(-150, 150),
    "tau_r": lim(-150, 150),
}
