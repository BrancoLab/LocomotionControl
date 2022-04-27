
import sys

sys.path.append("./")

# test the RL environment with random actions while making a video

from environment import MTMEnv
from utils import make_video

from train import make_env


env = make_env(1)( max_n_steps=5000)
env.reset()
make_video(None, env, video_length=200)