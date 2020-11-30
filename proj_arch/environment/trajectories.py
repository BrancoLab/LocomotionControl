import numpy as np
import pandas as pd

from fcutils.maths.geometry import (
    calc_angle_between_points_of_vector_2d,
    calc_distance_between_points_in_a_vector_2d,
    calc_ang_velocity,
)
from fcutils.maths.utils import derivative

from proj.utils.misc import interpolate_nans
from proj.environment.bezier import calc_bezier_path


# ------------------------------ From real data ------------------------------ #
def simulated_but_realistic(params, planning_params, cache_fld, *args):

    """
        Creates an artificial trajectory similar to the ones
        you'd get by loading trajectories from tracking data.
    """

    duration = np.random.uniform(1.5, 6)
    n_steps = int(duration / params["dt"])

    # Define start and end of traj
    p0 = np.array([0, 0])
    p1 = np.array([0, 80])

    # Define an additional random point
    while True:
        p2 = np.array(
            [
                np.random.uniform(low=-120, high=120),
                np.random.uniform(low=10, high=70),
            ]
        )

        if np.abs(p2[0]) > 50:
            break

    # Interpolate line segments
    xy = calc_bezier_path(np.vstack([p0, p2, p1]), params["n_steps"])
    x, y = xy[:, 0], xy[:, 1]

    # Get theta
    theta = calc_angle_between_points_of_vector_2d(x, y)
    theta = np.radians(90 - theta)
    theta = np.unwrap(theta)
    theta[0] = theta[1]

    # Get ang vel
    omega = calc_ang_velocity(theta)
    omega[0] = omega[2]
    omega[1] = omega[2]

    # Get speed
    v = calc_distance_between_points_in_a_vector_2d(x, y)
    v[0] = v[1]

    speedup_factor = params["n_steps"] / n_steps
    v *= speedup_factor
    v *= 1 / params["dt"]

    # stack
    trajectory = np.vstack([x, y, theta, v, omega]).T

    return (
        trajectory,
        duration,
        None,
    )


def from_tracking(params, planning_params, cache_fld, *args):
    """
        Get a trajectory from real tracking data, cleaning it up
        a little in the process.
    """
    # Keep only trials with enough frames
    trials = pd.read_hdf(cache_fld, key="hdf")

    # select a single trials
    if not params["randomize"]:
        trial = trials.iloc[params["trial_n"]]
    else:
        trial = trials.sample().iloc[0]

    # Get variables
    fps = trial.fps
    x = trial.x * params["px_to_cm"]
    y = trial.y * params["px_to_cm"]

    angle = interpolate_nans(trial.orientation)
    angle = np.radians(90 - angle)
    angle = np.unwrap(angle)

    speed = trial.speed * fps * params["px_to_cm"]
    ang_speed = derivative(angle)

    # resample variables so that samples are uniformly distributed
    vrs = (x, y, angle, speed, ang_speed)
    if params["resample"]:
        t = np.arange(len(x))
        vrs = [
            calc_bezier_path(np.vstack([t, v]).T[:, 1], params["n_steps"])
            for v in vrs
        ]

    # stack
    trajectory = np.vstack(vrs).T
    return (
        trajectory,
        len(x) / fps,
        trial,
    )
