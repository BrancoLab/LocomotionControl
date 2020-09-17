import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from rich.table import Table
from rich import print

from fcutils.maths.geometry import (
    calc_angle_between_points_of_vector_2d,
    calc_distance_between_points_in_a_vector_2d,
    calc_distance_between_points_2d,
)
from fcutils.maths.filtering import line_smoother

from proj.utils import interpolate_nans


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
    return compute_trajectory_stats(
        trajectory, len(trajectory), planning_params
    )


def compute_trajectory_stats(trajectory, duration, planning_params):
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

    print("\n\n")
    print(table)

    if waypoint_density < 2 or waypoint_density > 3:
        print(
            f"[bold red]Waypoint density of {round(waypoint_density, 3)} out of range, it should be 2 < wp < 3!"
        )

    if perc_lookahead < 0.05:
        print(
            f"[bold red]Lookahead of {lookahead} is {perc_lookahead} of the # of waypoints, that might be too low. Values closer to 5% are advised."
        )

    return trajectory, duration


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


def from_tracking(
    n_steps, params, planning_params, cache_fld, *args, skip=100
):
    if not params["randomize"]:
        trial = pd.read_hdf(cache_fld, key="hdf").iloc[0]
    else:
        trial = pd.read_hdf(cache_fld, key="hdf").sample().iloc[0]

    # Get variables
    x = trial.body_xy[:, 0]
    y = trial.body_xy[:, 1]

    angle = interpolate_nans(trial.body_orientation)
    angle = np.radians(90 - angle)
    angle = np.unwrap(angle)

    speed = line_smoother(trial.body_speed) + trial.fps

    ang_speed = np.ones_like(speed)  # it will be ignored

    # resample variables so that samples are uniformly distribued
    vars = dict(x=x, y=y, angle=angle, speed=speed, ang_speed=ang_speed)
    if params["resample"]:
        for k, v in vars.items():
            v = v[skip:-10]
            vars[k] = _interpol(v, params["max_deg_interpol"], n_steps)

    # stack and cut
    trajectory = np.vstack(vars.values()).T

    trajectory = trajectory[params["skip"] :, :]

    return compute_trajectory_stats(
        trajectory, len(x) / trial.fps, planning_params
    )
