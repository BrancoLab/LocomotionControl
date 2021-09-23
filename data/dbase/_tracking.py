from loguru import logger
import numpy as np
from scipy.signal import medfilt
import pandas as pd

from fcutils.maths.geometry import (
    calc_distance_between_points_in_a_vector_2d as get_speed_from_xy,
)
from fcutils.maths.geometry import (
    calc_angle_between_points_of_vector_2d as get_dir_of_mvmt_from_xy,
)
from fcutils.maths.geometry import (
    calc_angle_between_vectors_of_points_2d as get_orientation,
)
from fcutils.maths import derivative
from fcutils.path import size

from data.dbase.io import load_dlc_tracking
from data import data_utils


def register(x: np.ndarray, y: np.ndarray, M: np.ndarray):
    xy = np.vstack([x, y]).T

    # Prep vars
    m3d = np.append(M, np.zeros((1, 3)), 0)

    # affine transform to match model arena
    concat = np.ones((len(xy), 3))
    concat[:, :2] = xy
    registered = np.matmul(m3d, concat.T).T[:, :2]

    return registered[:, 0], registered[:, 1]


def process_body_part(
    bp_data: dict,
    M: np.ndarray,
    likelihood_th: float = 0.95,
    cm_per_px: float = 1,
) -> dict:
    # register to CMM
    x, y = register(bp_data["x"], bp_data["y"], M)

    # scale to go px -> cm
    x *= cm_per_px
    y *= cm_per_px

    # remove low confidence intervals
    like = bp_data["likelihood"]
    x[like < likelihood_th] = np.nan
    y[like < likelihood_th] = np.nan

    # interpolate nans
    xy = data_utils.interpolate_nans(x=x, y=y)
    x, y = np.array(list(xy["x"].values())), np.array(list(xy["y"].values()))

    # median filter pass
    x = data_utils.convolve_with_gaussian(x, kernel_width=11)
    y = data_utils.convolve_with_gaussian(y, kernel_width=11)

    # flip data on X axis
    x = x - np.min(x)
    x_mean = np.mean(x, axis=0)
    x = (x_mean - x) + x_mean

    # compute speed
    speed = (
        data_utils.convolve_with_gaussian(
            get_speed_from_xy(x, y), kernel_width=15
        )
        * 60
    )  # speed in cm / s
    speed[:5] = speed[6]
    speed[-5:] = speed[-6]

    # make sure there are no nans
    results = dict(x=x, y=y, bp_speed=speed)
    for k, var in results.items():
        if np.any(np.isnan(var)):
            raise ValueError(f"Found NANs in {k}")

    return results


def calc_angular_velocity(angles: np.ndarray) -> np.ndarray:
    # convert to radians and take derivative
    rad = np.unwrap(np.deg2rad(angles))
    rad = data_utils.convolve_with_gaussian(rad, 25)

    diff = derivative(rad)
    return np.rad2deg(diff)


def compute_averaged_quantities(body_parts_tracking: dict) -> dict:
    """
        For some things like orientation average across body parts to reduce noise
    """
    # import matplotlib.pyplot as plt
    def unwrap(x):
        return np.degrees(np.unwrap(np.radians(x)))

    # get data
    body = pd.DataFrame(body_parts_tracking["body"]).interpolate(axis=0)
    tail_base = pd.DataFrame(body_parts_tracking["tail_base"]).interpolate(
        axis=0
    )

    # get speed
    results = dict(speed=body.bp_speed.values.copy())
    results["speed"] = medfilt(results["speed"], 11)

    # get direction of movement
    results["direction_of_movement"] = get_dir_of_mvmt_from_xy(body.x, body.y)
    results["dmov_velocity"] = (
        calc_angular_velocity(results["direction_of_movement"]) * 60
    )

    results["dmov_velocity"] = data_utils.remove_outlier_values(
        results["dmov_velocity"],
        50,
        errors_calculation_array=np.abs(derivative(results["dmov_velocity"])),
    )

    results["direction_of_movement"][
        np.where(results["speed"] < 2)[0]
    ] = np.nan  # no dir of mvmt when there is no mvmt
    results["dmov_velocity"][
        np.where(results["speed"] < 2)[0]
    ] = np.nan  # no dir of mvmt when there is no mvmt

    # compute orientation of each body part
    results["orientation"] = get_orientation(
        tail_base.x, tail_base.y, body.x, body.y
    )

    # compute angular velocity in deg/s
    avel = calc_angular_velocity(results["orientation"]) * 60
    results["angular_velocity"] = data_utils.remove_outlier_values(
        avel.copy(), 600, errors_calculation_array=np.abs(avel)
    )

    return results


def process_tracking_data(
    key: dict,
    tracking_file: str,
    M: np.ndarray,
    likelihood_th: float = 0.95,
    cm_per_px: float = 1,
):
    def merge_two_dicts(x: dict, y: dict) -> dict:
        z = x.copy()  # start with keys and values of x
        z.update(y)  # modifies z with keys and values of y
        return z

    # load data
    logger.debug(
        f"Loading tracking data: {tracking_file.name} ({size(tracking_file)})"
    )
    body_parts_tracking: dict = load_dlc_tracking(tracking_file)

    # process each body part
    logger.debug("      processing body parts tracking data")
    body_parts_tracking = {
        k: process_body_part(
            bp, M, likelihood_th=likelihood_th, cm_per_px=cm_per_px
        )
        for k, bp in body_parts_tracking.items()
    }

    # compute orientation, angular velocity and speed
    logger.debug("      computing body orientation")
    velocites = compute_averaged_quantities(body_parts_tracking)

    # make sure all tracking start at (x,y)=(0, 0)
    x0 = np.nanmin(body_parts_tracking["body"]["x"])
    y0 = np.nanmin(body_parts_tracking["body"]["y"])

    # merge dictionaries
    body_parts_tracking = {
        bp: merge_two_dicts(data, key)
        for bp, data in body_parts_tracking.items()
    }
    for bp in body_parts_tracking.keys():
        body_parts_tracking[bp]["bpname"] = bp
        body_parts_tracking[bp]["x"] = body_parts_tracking[bp]["x"] - x0
        body_parts_tracking[bp]["y"] = body_parts_tracking[bp]["y"] - y0

    key.update(velocites)

    return key, body_parts_tracking, len(key["orientation"])


def get_movements(
    key: dict,
    tracking: pd.DataFrame,
    moving_threshold: float,
    turning_threshold: float,
) -> dict:
    """
        Creates array indicating when the mouse is doing different kinds of movements
    """
    base_array = np.zeros_like(tracking.x)

    # get when moving
    key["moving"] = base_array.copy()
    key["moving"][np.where(tracking.speed > moving_threshold)[0]] = 1

    # get when turning left
    key["turning_left"] = base_array.copy()
    key["turning_left"][
        np.where(tracking.angular_velocity > turning_threshold)[0]
    ] = 1

    # get when turning right
    key["turning_right"] = base_array.copy()
    key["turning_right"][
        np.where(tracking.angular_velocity < -turning_threshold)[0]
    ] = 1

    return key
