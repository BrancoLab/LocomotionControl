import sys

sys.path.append("./")

from pathlib import Path
import os
import numpy as np
import json
import glob
from loguru import logger

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from stable_baselines3 import TD3

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import torch as th

# from utils import make_video
from environment import MTMEnv
from utils import make_video


def make_env(rank, seed=0, log_dir=None, **kwargs):
    """
    Utility function for multiprocessed env.

    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init(**kwargs):
        env = MTMEnv(log_dir=log_dir, **kwargs)
        env.seed(seed + rank)
        log_file = (
            os.path.join(log_dir, str(rank)) if log_dir is not None else None
        )
        env = Monitor(env, log_file)
        return env

    set_random_seed(seed)
    return _init


def make_agent(
    env,
    tensorboard_log=None,
    params=dict(),
    policy_kwargs=dict(),
    action_noise_kwargs=None,
):
    if (
        action_noise_kwargs is not None
        and action_noise_kwargs["use_action_noise"]
    ):
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_kwargs["noise_std"] * np.ones(n_actions),
        )
    else:
        action_noise = None

    model = TD3(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        action_noise=action_noise,
        **params,
    )
    return model


class SaveVideoCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, save_freq: int, log_dir: str, verbose: int = 1):
        super(SaveVideoCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "videos")

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # get last .zip file in log_dir
            files = glob.glob(os.path.join(self.log_dir, "*.zip"))
            if not files:
                return True

            logger.info(f"Found {len(files)} .zip files in {self.log_dir}")

            steps = [int(Path(f).stem.split("_")[1]) for f in files]
            idx = np.argmax(steps)

            _file = sorted(files)[idx]
            n_iter = Path(_file).stem

            make_video(
                self.model,
                self.model.get_env(),
                os.path.join(self.save_path, f"{n_iter}.mp4"),
                video_length=25000,
            )

        return True


td3_params = dict(
    learning_rate=1e3,
    gamma=0.99,
    learning_starts=10000,
    buffer_size=200000,
    train_freq=1,
    gradient_steps=2,
    batch_size=256,
    verbose=1,
    device="auto",
)


if __name__ == "__main__":
    fld = Path(
        "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/RL"
    )

    # ---------------------------------- params ---------------------------------- #
    NAME = "TD32"
    N_CPU = 1
    N_STEPS = 100_000
    SEED = 0

    PARAMS = td3_params

    policy_kwargs = dict(
        # net_arch=[512, 512],
        net_arch=[128, 128],
        activation_fn=th.nn.ReLU,
    )

    action_noise_kwargs = dict(
        use_action_noise=True, noise_type="normal", noise_std=0.05
    )

    # -------------------------------- store logs -------------------------------- #
    log_dir = fld / NAME
    log_dir.mkdir(exist_ok=True)

    # get subdirs in log_dir to save in a separate one
    subdirs = [x for x in log_dir.iterdir() if x.is_dir()]
    log_dir = str(log_dir / ("rep_" + str(len(subdirs) + 1)))
    os.makedirs(log_dir, exist_ok=True)

    # save params to json
    with open(os.path.join(log_dir, "data.json"), "w") as fp:
        prms = {
            **PARAMS,
            **dict(ncpu=N_CPU, nsteps=N_STEPS, seed=SEED),
            **{k: str(v) for k, v in policy_kwargs.items()},
            **action_noise_kwargs,
        }
        json.dump(prms, fp, indent=4)

    # -------------------------------- create env -------------------------------- #
    if N_CPU > 1:
        env = DummyVecEnv(
            [make_env(i, log_dir=log_dir, seed=SEED) for i in range(N_CPU)]
        )  # or SubprocVecEnv/DummyVecEnv
    else:
        env = make_env(1, seed=SEED, log_dir=log_dir)()
        # check_env(env)

    # ------------------------------- model & train ------------------------------ #
    model = make_agent(env, tensorboard_log=log_dir, params=PARAMS)

    checkpoint_callback = CheckpointCallback(
        save_freq=2500, save_path=log_dir, name_prefix=NAME
    )
    # video_callback = SaveVideoCallback(25000, log_dir)
    # _cb = CallbackList([checkpoint_callback, video_callback])
    _cb = checkpoint_callback
    model.learn(total_timesteps=N_STEPS, callback=_cb, log_interval=10)

    model.save(os.path.join(log_dir, "trained_{NAME}"))
    logger.info("Done")

    # make video
    # video_name = os.path.join(log_dir, f"video_{NAME}.mp4")
    # make_video(model, env, video_name, video_length=150)
