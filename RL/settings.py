from collections import namedtuple

BUFFER_SIZE = 5 * int(1e4)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0.0  # L2 weight decays

N_EPISODES = 10000
MAX_EPISODE_LEN = 1000
MIN_GOAL_DISTANCE = 10

NOISE_SCALE = 0.2


# INPUTS SCALES
lim = namedtuple("lim", "min, max")
# max values for network's inputs
R_MAX = lim(-150, 150)
PSY_MAX = lim(-360, 360)
V_MAX = lim(-60, 60)
OMEGA_MAX = lim(-100, 100)
