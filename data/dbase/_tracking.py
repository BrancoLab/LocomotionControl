from loguru import logger
import numpy as np
from scipy.signal import medfilt
import pandas as pd
import json

from fcutils.maths import derivative
from fcutils.path import size, files

from data.dbase.io import load_dlc_tracking
from data import data_utils
from geometry.angles import orientation, angular_derivative
from geometry import Path
from data.paths import processed_tracking
from data.dbase._kallman_filtering import kalmann


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

    # use kalmann filtering to smooth XY tracking and estimate speed and accelerations
    res = kalmann(np.vstack([x, y]))

    # compute overall speed based on x/y speed components
    speed = np.sqrt(res["xdot"] ** 2 + res["ydot"] ** 2)  # already in cm/s

    # compute the angle of the velocity vector
    beta = np.degrees(np.arctan2(res["ydot"], res["xdot"]))

    # make sure there are no nans
    results = dict(
        x=x, y=y, bp_speed=speed, xdot=res["xdot"], ydot=res["ydot"], beta=beta
    )
    for k, var in results.items():
        if np.any(np.isnan(var)):
            raise ValueError(f"Found NANs in {k}")

    return results


def calc_angular_velocity(angles: np.ndarray) -> np.ndarray:
    # convert to radians and take derivative
    rad = np.unwrap(np.deg2rad(angles))
    rad = medfilt(rad, 11)
    rad = data_utils.convolve_with_gaussian(rad, 11)

    diff = derivative(rad)
    return np.rad2deg(diff)


def compute_velocities(body_parts_tracking: dict) -> dict:
    """
        For some things like orientation average across body parts to reduce noise
    """
    # get data
    body = pd.DataFrame(body_parts_tracking["body"]).interpolate(axis=0)
    tail_base = pd.DataFrame(body_parts_tracking["tail_base"]).interpolate(
        axis=0
    )

    # get speed & acceleration
    results = dict(speed=body.bp_speed.values.copy())
    results["acceleration"] = derivative(results["speed"])

    # get longitudinal speed and acceleration

    # get direction of movement
    path = Path(body.x, body.y).smooth(window=3)
    results["theta"] = path.theta
    results["thetadot"] = path.thetadot  # deg/s
    results["thetadotdot"] = path.thetadotdot  # in deg / s^2

    # compute orientation of the body
    results["orientation"] = orientation(
        tail_base.x, tail_base.y, body.x, body.y, smooth=False
    )

    # compute angular velocity in deg/s
    results["angular_velocity"] = (
        angular_derivative(results["orientation"]) * 60
    )  # in deg/s

    return results


def _to_arr(v):
    if isinstance(v[0], str):
        return "".join(v)
    else:
        return np.array(v)


def load_processed_tracking(tracking_file):
    """
        Attempt to load a processed tracking file to avoid processing anew
    """
    files_found = files(processed_tracking)
    if files_found is None or not isinstance(files_found, list):
        return None, None

    processed = [f.stem for f in files_found]
    if tracking_file.stem in processed:
        logger.info(f"Loading processed tracking file: {tracking_file.stem}")

        _path = processed_tracking / (tracking_file.stem + ".json")
        # load from json file
        with open(_path, "r") as f:
            data = json.load(f)

        key = {k: _to_arr(v) for k, v in data["key"].items()}
        del data["key"]
        data = {
            k: {k2: _to_arr(v2) for k2, v2 in v.items()}
            for k, v in data.items()
        }
        return key, data
    else:
        return None, None


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

    # # try using processed tracking file
    # _key, _data = load_processed_tracking(tracking_file)
    # if _key is not None:
    #     return _key, _data, len(_key["orientation"])

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
    velocites = compute_velocities(body_parts_tracking)

    # remove extra keys in bodyparts data
    for k in body_parts_tracking.keys():
        del body_parts_tracking[k]["xdot"]
        del body_parts_tracking[k]["ydot"]

    # make sure all tracking start at (x,y)=(0, 0) and ends at (x,y)=(40, 60)
    limits = dict(
        x=(
            np.percentile(body_parts_tracking["body"]["x"], 0.001),
            np.percentile(body_parts_tracking["body"]["x"], 99.999),
            2,
            39,
        ),
        y=(
            np.percentile(body_parts_tracking["body"]["y"], 0.001),
            np.percentile(body_parts_tracking["body"]["y"], 99.999),
            1.5,
            57.5,
        ),
    )

    # merge dictionaries
    body_parts_tracking = {
        bp: merge_two_dicts(data, key)
        for bp, data in body_parts_tracking.items()
    }

    for bp in body_parts_tracking.keys():
        body_parts_tracking[bp]["bpname"] = bp
        for coord in "xy":
            at_zero = body_parts_tracking[bp][coord] - limits[coord][0]
            scaled = (
                at_zero
                / (limits[coord][1] - limits[coord][0])
                * (limits[coord][3] - limits[coord][2])
            )
            body_parts_tracking[bp][coord] = scaled + limits[coord][2]

    logger.info(
        f"""
        Coordinates bounds: 
            x: {np.min(body_parts_tracking['body']['x'])} -> {np.max(body_parts_tracking['body']['x'])}
            y: {np.min(body_parts_tracking['body']['y'])} -> {np.max(body_parts_tracking['body']['y'])}
        """
    )
    key.update(velocites)

    # save body_parts_tracking to .json file in the processed tracking data folder
    with open(processed_tracking / (tracking_file.stem + ".json"), "w") as f:
        _data = body_parts_tracking.copy()
        _data["key"] = key
        _data = {
            k: {kk: list(vv) for kk, vv in v.items()} for k, v in _data.items()
        }
        json.dump(_data, f)

    return key, body_parts_tracking, len(key["orientation"])


def get_movements(
    key: dict,
    tracking: pd.DataFrame,
    moving_threshold: float,
    turning_threshold: float,
    paw_speed_th=10,
) -> dict:
    """
        Creates array indicating when the mouse is doing different kinds of movements
    """

    paws = tracking.loc[
        tracking.bpname.isin(["left_fl", "left_hl", "right_fl", "right_hl"])
    ]

    # get when each paw is moving
    paws_moving = {
        f"{p.bpname}_moving": np.int64(p.speed >= paw_speed_th)
        for i, p in paws.iterrows()
    }
    paws_moving["walking"] = np.prod(
        np.vstack(list(paws_moving.values())), axis=0
    )

    body = tracking.loc[tracking.bpname == "body"].iloc[0]
    base_array = np.zeros_like(body.x)

    # get when moving
    key["moving"] = base_array.copy()
    key["moving"][np.where(body.speed > moving_threshold)[0]] = 1

    # get when turning left
    key["turning_left"] = base_array.copy()
    key["turning_left"][
        np.where(body.angular_velocity > turning_threshold)[0]
    ] = 1

    # get when turning right
    key["turning_right"] = base_array.copy()
    key["turning_right"][
        np.where(body.angular_velocity < -turning_threshold)[0]
    ] = 1

    key = {**key, **paws_moving}

    return key
