import numpy as np
import pandas as pd
from loguru import logger

from fcutils.maths.geometry import (
    calc_angle_between_points_of_vector_2d,
    calc_distance_between_points_in_a_vector_2d,
)
from fcutils.maths.utils import derivative

from .utils import interpolate_nans, calc_bezier_path, pol2cart
from .config import dt, px_to_cm, TRAJECTORY_CONFIG


# ------------------------------ From real data ------------------------------ #
def simulated():
    """
        Creates an artificial trajectory similar to the ones
        you'd get by loading trajectories from tracking data.

        To define a trajectory of N points:
            1. start with a point (origin at first)
            2. draw an angle and a distance from uniform distributions
            3. turn these into cartesian coordinates and add to previous
                point's coordiates
            4. this is the coordiates for the next point

        in practice two nearby points are drawn at each cycle to ensure steeper bends

        The finally compute the bezier path across all these points
    """

    duration = 5  # np.random.uniform(1.5, 6)
    n_steps = int(duration / dt)

    logger.info(
        f"Making simulated traj with duration: {duration} and n steps: {n_steps}"
    )

    # ? make simulated trajectory of N points
    # first point is origin
    points = [np.array([0, 0])]

    for n in range(100):
        # draw random angle and
        phi = np.random.uniform(90, 160) * (
            -1 if np.random.rand() < 0.5 else 1
        )
        if n == 0:
            phi = np.random.uniform(0, 360)

        phi = 0

        rho = np.random.uniform(10, 25)

        # get next two points coordinates
        previous = points[-1]
        for i in range(2):
            if i == 1:
                rho += rho / 2
                phi = phi / 2
            nxt = np.array(pol2cart(rho, phi)) + previous

            # append to list
            points.append(nxt)

    # Interpolate line segments
    xy = calc_bezier_path(np.vstack(points), TRAJECTORY_CONFIG["n_steps"])
    x, y = xy[:, 0], xy[:, 1]

    # Get theta
    theta = calc_angle_between_points_of_vector_2d(x, y)
    theta = np.radians(90 - theta)
    theta = np.unwrap(theta)
    theta[0] = theta[1]

    # Get ang vel
    speedup_factor = TRAJECTORY_CONFIG["n_steps"] / n_steps

    omega = derivative(theta)
    omega[0] = omega[1]
    omega *= 1 / dt

    # Get speed
    v = calc_distance_between_points_in_a_vector_2d(x, y)
    # v = np.linspace(50, 100, len(x))
    logger.info(
        f"Simulated trajectory total distance: {np.sum(np.abs(v)):.3f}, total angle: {np.sum(np.abs(np.degrees(omega))):.3f}"
    )

    # adjust speed
    v *= speedup_factor
    v *= 1 / dt

    # get 0 vectors for tau
    zeros = np.zeros_like(v) * 0.001  # to avoid 0 in divisions

    # stack
    trajectory = np.vstack([x, y, theta, v, omega, zeros, zeros]).T
    trajectory = trajectory[200:-200, :]

    return (
        trajectory,
        duration,
        None,
    )


def from_tracking(cache_fld, trialn=None):
    """
        Get a trajectory from real tracking data, cleaning it up
        a little in the process.
    """

    # Get a trial
    trials = pd.read_hdf(cache_fld, key="hdf")
    if trialn is None:
        trial = trials.sample().iloc[0]
    else:
        trial = trials.iloc[trialn]

    # Get variables
    fps = trial.fps
    x = trial.x * px_to_cm
    y = trial.y * px_to_cm

    angle = interpolate_nans(trial.orientation)
    angle = np.radians(90 - angle)
    angle = np.unwrap(angle)

    speed = trial.speed * fps * px_to_cm
    ang_speed = derivative(angle)

    # resample variables so that samples are uniformly distributed
    vrs = (x, y, angle, speed, ang_speed)
    t = np.arange(len(x))
    vrs = [
        calc_bezier_path(
            np.vstack([t, v]).T[:, 1], TRAJECTORY_CONFIG["n_steps"]
        )
        for v in vrs
    ]

    # get 0 vectors for tau
    # zeros = np.zeros_like(v) * 0.001  # to avoid 0 in divisions

    raise NotImplementedError("Add 0s")

    # stack
    trajectory = np.vstack(vrs).T
    return (
        trajectory,
        len(x) / fps,
        trial,
    )
