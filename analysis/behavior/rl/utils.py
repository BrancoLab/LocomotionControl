import matplotlib.pyplot as plt
from celluloid import Camera
from loguru import logger
from rich.progress import track

def make_video(model, env, video_name="video.mp4", video_length=150):

    # try:
    #     _env = env.envs[-1].env
    #     while _env.__class__.__name__ != "Environment":
    #         _env = _env.env
    # except:
    #     _env = env.unwrapped.env
    _env = env.envs[0].env
    camera = Camera(_env.fig)


    logger.info(f"Writing video to {video_name}")

    # Record the video starting at the first step
    obs = env.reset()
    rew = 0.0
    for i in track(range(video_length + 1), description="Recording video", total=video_length + 1):
        _env.render()

        # add text to frame with reward value
        rew = rew if isinstance(rew, (float, int)) else rew[0]
        _env.ax.set(title=f"Reward = {rew:.2f}")

        # capture frame
        camera.snap()

        # execute next action
        # action = env.action_space.sample()
        action = model.predict(obs)
        try:
            obs, rew, _, _ = env.step(action)
        except:
            # logger.debug("Error in step during video creation")
            break

    # Save the video
    animation = camera.animate()
    animation.save(video_name, fps=int(1/_env.dt))
    logger.info("Done & saved")

def inbounds(var, low, high):
    return min(high, max(var, low))


def normalize_to_unitbox(v, a, b):
    """
        Normalize a value v in [a, b] to the [-1, 1]
        unit box
    """
    return 2 * (v - a) / (b - a) - 1


def unnormalize(v, a, b):
    """
        Take a value v in [-1, 1] and unnormalize it to
        the original range [a, b]
    """
    return (v + 1) / 2 * (b - a) + a

