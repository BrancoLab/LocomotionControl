import numpy as np
from pathlib import Path
import pandas as pd

from fcutils.file_io.io import load_yaml


def load_results_from_folder(folder):
    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"Results folder {folder} doesnt exist")

    files = dict(
        config=folder / "config.yml",
        control=folder / "control_vars.yml",
        state=folder / "state_vars.yml",
        trajectory=folder / "trajectory.npy",
        history=folder / "history.h5",
    )

    for f in files.values():
        if not f.exists():
            raise ValueError(
                f"Data folder incomplete, something missing in : {str(folder)}.\n {f} is missing"
            )

    config = load_yaml(str(files["config"]))
    control = load_yaml(str(files["control"]))
    state = load_yaml(str(files["state"]))
    trajectory = np.load(str(files["trajectory"]))
    history = pd.read_hdf(str(files["history"]), key="hdf")

    return config, control, state, trajectory, history


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
