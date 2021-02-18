import numpy as np
from scipy.special import comb
from scipy import interpolate

from fcutils.maths.signals import derivative, rolling_mean
from fcutils.maths.geometry import (
    calc_angle_between_points_of_vector_2d as get_theta_from_xy,
)


def get_theta_omega_from_xy(x, y, dt=1, smooth=False):
    """
        Returns the orientation (direction of movement to be exact)
        and angular velocity given an XY trajectory

        Arguments:
            x, y: 1d np.ndarray with coordinates
            dt: float time interval betwen frames
    """

    if smooth:
        x = rolling_mean(x, 100)
        y = rolling_mean(y, 100)

    # Get theta
    theta = get_theta_from_xy(x, y)
    theta[0] = theta[1]
    theta = 90 - theta
    theta = np.unwrap(np.radians(theta))

    # Get ang vel
    omega = derivative(theta)
    omega[0] = omega[1]
    omega[-1] = omega[-2]
    omega *= 1 / dt

    return theta, omega


def merge(*ds):
    """
        Merges an arbitrary number of dicts or named tuples
    """
    res = {}
    for d in ds:
        if not isinstance(d, dict):
            res = {**res, **d._asdict()}
        else:
            res = {**res, **d}
    return res


def interpolate_nans(arr):
    """
    interpolate to fill nan values
    """
    inds = np.arange(arr.shape[0])
    good = np.where(np.isfinite(arr))
    f = interpolate.interp1d(inds[good], arr[good], bounds_error=False)
    B = np.where(np.isfinite(arr), arr, f(inds))
    return B


def calc_bezier_path(control_points, n_points=100):
    def Comb(n, i, t):
        return comb(n, i) * t ** i * (1 - t) ** (n - i)

    def bezier(t, control_points):
        n = len(control_points) - 1
        return np.sum(
            [Comb(n, i, t) * control_points[i] for i in range(n + 1)], axis=0
        )

    traj = []

    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))

    return np.array(traj)


def fit_angle_in_range(
    angles, min_angle=-np.pi, max_angle=np.pi, is_deg=False
):
    """ Check angle range and correct the range
    it assumes that the angles are passed ind degrees
    
    Args:
        angle (numpy.ndarray): in radians
        min_angle (float): maximum of range in radians, default -pi
        max_angle (float): minimum of range in radians, default pi
    Returns: 
        fitted_angle (numpy.ndarray): range angle in radians
    """
    if max_angle < min_angle:
        raise ValueError("max angle must be greater than min angle")
    if (max_angle - min_angle) < 2.0 * np.pi:
        raise ValueError(
            "difference between max_angle \
                          and min_angle must be greater than 2.0 * pi"
        )
    if is_deg:
        output = np.radians(angles)
    else:
        output = np.array(angles)

    output_shape = output.shape

    output = output.flatten()
    output -= min_angle
    output %= 2 * np.pi
    output += 2 * np.pi
    output %= 2 * np.pi
    output += min_angle

    output = np.minimum(max_angle, np.maximum(min_angle, output))
    output = output.reshape(output_shape)

    # if is_deg:
    #     output = np.degrees(output)
    return output
