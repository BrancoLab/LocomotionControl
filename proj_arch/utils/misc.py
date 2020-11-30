import numpy as np
from scipy import interpolate
from pathlib import Path
import pandas as pd
import time
import pickle

from fcutils.file_io.io import load_yaml
from fcutils.maths.geometry import calc_distance_from_point


def trajectory_at_each_simulation_step(trajectory, history):
    return np.vstack([trajectory[i, :] for i in history.trajectory_idx])


# ---------------------------------- Data IO --------------------------------- #


def load_results_from_folder(folder):
    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"Results folder {folder} doesnt exist")

    files = dict(
        config=folder / "config.yml",
        trajectory=folder / "init_trajectory.npy",
        history=folder / "history.h5",
        cost_history=folder / "cost_history.h5",
        trial=folder / "trial.h5",
        info=folder / "info.pkl",
    )

    for f in files.values():
        if not f.exists():
            if "trial.h5" in str(f):
                continue
            raise ValueError(
                f"Data folder incomplete, something missing in : {str(folder)}.\n {f} is missing"
            )

    config = load_yaml(str(files["config"]))
    trajectory = np.load(str(files["trajectory"]))
    history = pd.read_hdf(str(files["history"]), key="hdf")
    cost_history = pd.read_hdf(str(files["cost_history"]), key="hdf")

    try:
        trial = pd.read_hdf(str(files["trial"]))
    except Exception:
        trial = None

    with open(str(files["info"]), "rb") as f:
        info = pickle.load(f)

    return config, trajectory, history, cost_history, trial, info


# -------------------------------- Coordinates ------------------------------- #


def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    gamma = np.arctan2(y, x)
    return r, gamma


def polar_to_cartesian(r, gamma):
    x = r * np.cos(gamma)
    y = r * np.sin(gamma)
    return x, y


def traj_to_polar(traj):
    """ 
        Takes a trjectory expressed as (x,y,theta,v,s)
        and converts it to (r, gamma, v, s)
    """

    new_traj = np.zeros((len(traj), 4))

    new_traj[:, 0] = calc_distance_from_point(traj[:, :2], [0, 0])

    new_traj[:, 1] = np.arctan2(traj[:, 1], traj[:, 0])

    return new_traj


# ----------------------------------- Misc ----------------------------------- #
def duration_from_history(history, config):
    nframes = len(history)
    return round(nframes * config["dt"], 3)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print(f"{method.__name__}  {round( (te - ts) * 1000, 2)} ms")
        return result

    return timed


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


def wrap_angle(angles):
    """ 
        Maps a list of angles in RADIANS to [-pi, pi]
    """
    angles = np.array(angles)
    return (angles + np.pi) % (2 * np.pi) - np.pi


def interpolate_nans(arr):
    """
    interpolate to fill nan values
    """
    inds = np.arange(arr.shape[0])
    good = np.where(np.isfinite(arr))
    f = interpolate.interp1d(inds[good], arr[good], bounds_error=False)
    B = np.where(np.isfinite(arr), arr, f(inds))
    return B


def timestamp():
    return time.strftime("%y%m%d_%H%M%S")
