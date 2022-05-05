
import sys

sys.path.append("./")

# test the RL environment with random actions while making a video

from environment import MTMEnv
from utils import make_video

from train import make_env, make_agent, td3_params


env = make_env(1)( max_n_steps=5000)
env.reset()

agent = make_agent(env, params=td3_params)
make_video(agent, env, video_length=600)