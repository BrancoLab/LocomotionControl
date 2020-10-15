import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from rich.table import Table
import logging
from rich import print

from fcutils.maths.geometry import (
    calc_angle_between_points_of_vector_2d,
    calc_distance_between_points_in_a_vector_2d,
    calc_distance_between_points_2d,
    calc_distance_from_point,
    calc_ang_velocity,
)
from fcutils.maths.filtering import line_smoother

from proj.utils.misc import interpolate_nans

# from proj import log
from loguru import logger


def complete_given_xy(x, y, params, planning_params):
    # Compute other variables that figure in the state vector
    angle = np.radians(90 - calc_angle_between_points_of_vector_2d(x, y))
    angle = np.unwrap(angle)

    speed = 1 - np.sin(np.linspace(0, 6, len(x)))
    speed = (
        speed * (params["max_speed"] - params["min_speed"])
        + params["min_speed"]
    )

    ang_speed = np.ones_like(speed)  # it will be ignored

    trajectory = np.vstack([x, y, angle, speed, ang_speed]).T
    return (
        compute_trajectory_stats(
            trajectory, len(trajectory), params, planning_params
        ),
        None,
    )


def compute_trajectory_stats(
    trajectory,
    duration,
    params,
    planning_params,
    min_dist_travelled=150,
    mute=False,
):

    # Compute stats
    n_points = len(trajectory)
    distance_travelled = np.sum(
        calc_distance_between_points_in_a_vector_2d(
            trajectory[:, 0], trajectory[:, 1]
        )
    )
    start_goal_distance = calc_distance_between_points_2d(
        trajectory[0, :2], trajectory[-1, :2]
    )

    waypoint_density = n_points / distance_travelled

    # Look into planning params
    lookahead = planning_params["prediction_length"]
    perc_lookahead = round(lookahead / n_points, 2)

    # Print stats
    table = Table(
        show_header=True, header_style="bold magenta", title="Trajectory stats"
    )
    table.add_column("# waypoints", width=12, style="green")
    table.add_column("dist. travelled", style="magenta")
    table.add_column("waypoints density", style="green")
    table.add_column("start -> goal distance", style="white")
    table.add_column("duration", style="white")
    table.add_column("# lookahead", style="green")
    table.add_column("proportion lookahead", style="green")

    table.add_row(
        str(n_points),
        str(round(distance_travelled, 2)),
        str(round(waypoint_density, 2)),
        str(round(start_goal_distance, 2)),
        str(round(duration, 2)),
        str(lookahead),
        str(perc_lookahead),
    )

    # log stuff
    if not mute:
        logger.info(
            f"""
            Trajectory metadata 


            n_points = {n_points}
            distance_travelled = {round(distance_travelled, 2)}
            waypoint_density = {round(waypoint_density, 2)}
            start_goal_distance = {round(start_goal_distance, 2)}
            duration = {round(duration, 2)}
            lookahead = {lookahead}
            perc_lookahead = {perc_lookahead}
        """
        )

        print(table)

    metadata = dict(
        n_points=n_points,
        distance_travelled=round(distance_travelled, 2),
        waypoint_density=round(waypoint_density, 2),
        start_goal_distance=round(start_goal_distance, 2),
        duration=round(duration, 2),
        lookahead=lookahead,
        perc_lookahead=perc_lookahead,
    )

    if not mute:
        if waypoint_density < 2 or waypoint_density > 3:
            logger.info(
                f"Waypoint density of {round(waypoint_density, 3)} out of range, it should be 2 < wp < 3!",
                extra={"markdown": True},
            )
            print(
                f"[bold red]Waypoint density of {round(waypoint_density, 3)} out of range, it should be 2 < wp < 3!"
            )

        if perc_lookahead < 0.05:
            logger.info(
                f"Lookahead of {lookahead} is {perc_lookahead} of the # of waypoints, that might be too low. Values closer to 5% are advised.",
                extra={"markdown": True},
            )
            print(
                f"[bold red]Lookahead of {lookahead} is {perc_lookahead} of the # of waypoints, that might be too low. Values closer to 5% are advised."
            )
        if distance_travelled < min_dist_travelled * params["px_to_cm"]:
            logger.warning(
                "Distance travelled below minimal requirement, erroring"
            )
            return None, None

    return trajectory, duration, metadata


# ---------------------------------- Curves ---------------------------------- #
def point(n_steps, params, planning_params, *akrgs):
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)

    return complete_given_xy(x, y, params, planning_params)


def line(n_steps, params, planning_params, *akrgs):
    y = np.linspace(0, params["distance"], n_steps)
    x = np.zeros_like(y)

    return complete_given_xy(x, y, params, planning_params)


def circle(n_steps, params, planning_params, *akrgs):
    p = np.linspace(0, 2 * np.pi, n_steps)
    r = params["distance"] / 2

    x = np.cos(p) * r
    y = np.sin(p) * r

    return complete_given_xy(x, y, params, planning_params)


def sin(n_steps, params, planning_params, *akrgs):
    x = np.linspace(0, params["distance"], n_steps)
    y = 5 * np.sin(0.1 * x)

    return complete_given_xy(x, y, params, planning_params)


def parabola(n_steps, params, planning_params, *akrgs):
    def curve(x, a, b, c):
        return (a * (x - b) ** 2) + +c

    # Define 3 points
    X = [0, params["distance"] / 2, params["distance"]]
    Y = [0, params["distance"] / 4, 0]
    #
    # fit curve and make trace
    coef, _ = curve_fit(curve, X, Y)

    y = np.linspace(0, params["distance"], n_steps)
    x = curve(y, *coef)

    return complete_given_xy(x, y, params, planning_params)


# ------------------------------ From real data ------------------------------ #
def _interpol(x, max_deg, n_steps):
    """
        interpolates x with a number of polynomials to find
        the one with the best fit
    """
    l = np.arange(len(x))
    l2 = np.linspace(0, len(x), n_steps)

    # try a bunch of degrees
    errs = []
    for deg in range(3, max_deg):
        f = np.poly1d(np.polyfit(l, x, deg))
        newx = f(l)
        errs.append(mean_squared_error(newx, x))

    best_deg = np.argmin(errs) + 3
    f = np.poly1d(np.polyfit(l, x, best_deg))
    return f(l2)


def simulated_but_realistic(
    n_steps, params, planning_params, cache_fld, *args
):
    duration = np.random.uniform(1.5, 4)
    n_simulation_steps = int(duration / params["dt"])

    # Define start and end of traj
    p0 = np.array([0, 0])
    p1 = np.array([0, 100])

    # Define an additional random point
    p2 = np.array(
        [
            np.random.uniform(low=-120, high=120),
            np.random.uniform(low=20, high=80),
        ]
    )

    # Get distance of each segment
    d1 = np.sqrt((p0[0] - p2[0]) ** 2 + (p0[1] - p2[1]) ** 2)
    d2 = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    d = d1 + d2

    # Get the number of points in each segment
    n1 = int((n_steps * d1) / d)
    n2 = int((n_steps * d2) / d)

    # Generate line segments
    segment1 = np.array(
        [np.linspace(p0[0], p2[0], n1), np.linspace(p0[1], p2[1], n1)]
    )
    segment2 = np.array(
        [np.linspace(p2[0], p1[0], n2), np.linspace(p2[1], p1[1], n2)]
    )

    # Interpolate line segments
    xy = np.hstack([segment1, segment2])
    x, y = xy[0, :], xy[1, :]

    x = _interpol(x, 5, n_steps)
    y = _interpol(y, 5, n_steps)

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

    speedup_factor = n_steps / n_simulation_steps
    v *= speedup_factor
    v *= 1 / params["dt"]

    # stack
    trajectory = np.vstack([x, y, theta, v, omega]).T

    return (
        compute_trajectory_stats(
            trajectory, duration, params, planning_params
        )[:2],
        None,
    )


def from_tracking(n_steps, params, planning_params, cache_fld, *args):
    # Keep only trials with enough frames
    trials = pd.read_hdf(cache_fld, key="hdf")
    trials["length"] = [len(t.body_xy) for i, t in trials.iterrows()]
    trials = trials.loc[
        trials.length > 50
    ]  # at least N frames in tracking data

    # select a single trials
    if not params["randomize"]:
        trial = trials.iloc[0]
    else:
        trial = trials.sample().iloc[0]

    # Get variables
    try:
        fps = trial.fps
    except:
        fps = 60

    x = trial.body_xy[:, 0] * params["px_to_cm"]
    y = trial.body_xy[:, 1] * params["px_to_cm"]

    angle = interpolate_nans(trial.body_orientation)
    angle = np.radians(90 - angle)
    angle = np.unwrap(angle)

    speed = line_smoother(trial.body_speed) * fps
    speed *= params["px_to_cm"]
    ang_speed = np.ones_like(speed)  # it will be ignored

    # get start frame
    from_start = calc_distance_from_point(np.array([x, y]), [x[0], y[0]])
    start = np.where(from_start > params["dist_th"])[0][0]

    if start >= len(x):
        logging.error("Bad trajectory")
        raise ValueError("Bad trajectory")

    # resample variables so that samples are uniformly distributed
    vars = dict(x=x, y=y, angle=angle, speed=speed, ang_speed=ang_speed)
    if params["resample"]:
        for k, v in vars.items():
            try:
                vars[k] = _interpol(
                    v[start:-1], params["max_deg_interpol"], n_steps
                )
            except ValueError:  # there were nans in the array
                return None, None

    # stack
    trajectory = np.vstack(vars.values()).T

    return (
        compute_trajectory_stats(
            trajectory, len(x[start:]) / fps, params, planning_params
        )[:2],
        trial,
    )
