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
from fcutils.maths.geometry import calc_ang_velocity
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

    # median filter pass
    x = medfilt(x, kernel_size=7)
    y = medfilt(y, kernel_size=7)

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

    # compute speed and direction of movement
    dir_of_mvmt = get_dir_of_mvmt_from_xy(x, y)
    speed = (
        medfilt(get_speed_from_xy(x, y), kernel_size=15) * 60
    )  # speed in cm / s

    # makre sure there are no nans
    results = dict(x=x, y=y, speed=speed, direction_of_movement=dir_of_mvmt)
    for k, var in results.items():
        if np.any(np.isnan(var)):
            raise ValueError(f"Found NANs in {k}")
    return results


def compute_body_orientation(body_parts_tracking: dict):
    # get data
    body = pd.DataFrame(body_parts_tracking["body"]).interpolate(azis=0)
    snout = pd.DataFrame(body_parts_tracking["snout"]).interpolate(azis=0)
    tail_base = pd.DataFrame(body_parts_tracking["tail_base"]).interpolate(
        azis=0
    )
     
    # compute orientation
    orientation_body = get_orientation(tail_base.x, tail_base.y, snout.x, snout.y)
    orientation_snout = get_orientation(body.x, body.y, snout.x, snout.y)
    orientation_tail = get_orientation(
        tail_base.x, tail_base.y, body.x, body.y
    )

    orientation =np.mean(
        np.vstack([orientation_body, orientation_snout, orientation_tail]), axis=0
    )


    # compute angular velocity
    avel = medfilt(np.mean(
        np.vstack([
            calc_ang_velocity(orientation_body),
            calc_ang_velocity(orientation_snout),
            calc_ang_velocity(orientation_tail),
        ]), axis=0
    ), kernel_size=15) * 60

    return orientation, avel


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

    # make sure all tracking start at (x,y)=(0, 0)
    x0 = np.nanmin(body_parts_tracking["body"]["x"])
    y0 = np.nanmin(body_parts_tracking["body"]["y"])

    # compute orientation
    logger.debug("      computing body orientation")
    orientation, angular_velocity = compute_body_orientation(
        body_parts_tracking
    )

    # merge dictionaries
    body_parts_tracking = {
        bp: merge_two_dicts(data, key)
        for bp, data in body_parts_tracking.items()
    }
    for bp in body_parts_tracking.keys():
        body_parts_tracking[bp]["bpname"] = bp
        body_parts_tracking[bp]["x"] = body_parts_tracking[bp]["x"] - x0
        body_parts_tracking[bp]["y"] = body_parts_tracking[bp]["y"] - y0

    key["orientation"] = orientation
    key["angular_velocity"] = angular_velocity

    # do a few checks
    if len(orientation) != len(body_parts_tracking["body"]["x"]):
        raise ValueError("Incoherent number of frames between data suorces")

    return key, body_parts_tracking, len(orientation)


def get_movements(key:dict, tracking:pd.DataFrame, moving_threshold:float, turning_threshold:float) -> dict:
    '''
        Creates array indicating when the mouse is doing different kinds of movements
    '''
    base_array = np.zeros_like(tracking.x)

    # get when moving
    key['moving'] = base_array.copy()
    key['moving'][np.where(tracking.speed > moving_threshold)[0]] = 1

    # get when turning left
    key['turning_left'] = base_array.copy()
    key['turning_left'][np.where(tracking.angular_velocity > turning_threshold)[0]] = 1

    # get when turning right
    key['turning_right'] = base_array.copy()
    key['turning_right'][np.where(tracking.angular_velocity < -turning_threshold)[0]] = 1

    return key