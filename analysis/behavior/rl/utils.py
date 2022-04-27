import cv2
from loguru import logger
from rich.progress import track


def make_video(model, env, video_name="video.mp4", video_length=50):

    # try:
    #     _env = env.envs[-1].env
    #     while _env.__class__.__name__ != "Environment":
    #         _env = _env.env
    # except:
    #     _env = env.unwrapped.env
    try:
        _env = env.envs[0].env
    except:
        _env = env
    _env.MAX_N_STEPS = video_length

    # setup frame
    logger.info(f"Writing video to {video_name}")
    frame = _env.render()[:, :, :-1]
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width, _ = frame.shape

    # crate a cv2.VideoWriter for a greyscale video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)

    # Record the video starting at the first step
    obs = env.reset()
    rew, action = 0.0, "ff"
    for i in track(range(video_length + 1), description="Recording video", total=video_length + 1):
        
        # capture frame 
        if i % 40 == 0:
            frame = _env.render()[:, :, :-1]

            # convert frame to grayscale
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # write frame to video
            cv2.imshow(video_name, frame)
            cv2.waitKey(10)
            video.write(frame)
        

        # execute next action
        # action = env.action_space.sample()
        # action = (action, None)
        action = model.predict(obs)
        try:
            obs, rew, done, _ = env.step(action)
        except Exception as e:
            logger.warning(f"Error in step during video creation: {e}")
            break

        if done:
            logger.info("Env says done")
            break

    # Save the video
    video.release()
    env.close()
    logger.info("Done & saved")

    # Close the window
    cv2.destroyAllWindows()


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

