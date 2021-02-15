BUFFER_SIZE = 5 * int(1e3)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0.0  # L2 weight decays

N_EPISODES = 10000
MAX_EPISODE_LEN = 1000
MIN_GOAL_DISTANCE = 10

NOISE_SCALE = 0.1
