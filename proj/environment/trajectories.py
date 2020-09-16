import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d
from fcutils.maths.filtering import line_smoother

from proj.utils import interpolate_nans
from proj.paths import trials_cache


def complete_given_xy(x, y, params):
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
    return trajectory[2:, :]


# ---------------------------------- Curves ---------------------------------- #
def point(n_steps, params):
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)

    return complete_given_xy(x, y, params)


def line(n_steps, params):
    y = np.linspace(0, params["distance"], n_steps)
    x = np.zeros_like(y)

    return complete_given_xy(x, y, params)


def circle(n_steps, params):
    p = np.linspace(0, 2 * np.pi, n_steps)
    r = params["distance"] / 2

    x = np.cos(p) * r
    y = np.sin(p) * r

    return complete_given_xy(x, y, params)


def sin(n_steps, params):
    x = np.linspace(0, params["distance"], n_steps)
    y = 5 * np.sin(0.1 * x)

    return complete_given_xy(x, y, params)


def parabola(n_steps, params):
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

    return complete_given_xy(x, y, params)


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


def from_tracking(n_steps, params, skip=135):
    trial = pd.read_hdf(trials_cache, key="hdf").iloc[0]

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

    return trajectory, len(x) / trial.fps
