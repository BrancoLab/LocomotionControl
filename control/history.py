import numpy as np
from pathlib import Path
import pandas as pd
import pickle

from .config import dt


def load_results_from_folder(folder):
    # load info
    with open(folder / "info.pkl", "rb") as f:
        info = pickle.load(f)

    # load history
    history = pd.read_hdf(folder / "history.h5", key="hdf")

    # Load trajectory
    traj = np.load(folder / "trajectory.npy")

    # load trial
    try:
        trial = pd.read_hdf(folder / "trial.h5", key="hdf")
    except Exception:
        trial = None

    return history, info, traj, trial


class History:
    def __init__(self):
        self.record = dict(
            x=[],
            y=[],
            theta=[],
            v=[],
            omega=[],
            tau_l=[],
            tau_r=[],
            trajectory_idx=[],
            nudot_left=[],
            nudot_right=[],
            goal_x=[],
            goal_y=[],
            goal_theta=[],
            goal_v=[],
            goal_omega=[],
        )

        self.info = dict()

    def append(self, *datas):
        for data in datas:
            if not isinstance(data, dict):
                data = data._asdict()

            for k, v in data.items():
                self.record[k].append(v)

    def save(self, folder, trajectory, trial):
        # Save info
        self.info["duration"] = len(self.record["x"]) / dt
        with open(str(folder / "info.pkl"), "wb") as out:
            pickle.dump(self.info, out)

        # save history
        pd.DataFrame(self.record).to_hdf(folder / "history.h5", key="hdf")

        # save trajectory
        np.save(
            str(folder / "trajectory.npy"), trajectory,
        )

        # save original trial
        if trial is not None:
            trial.to_hdf(folder / "trial.h5", key="hdf")

        # save configs
        with open(folder / "config.txt", "w") as out:
            with open(Path(__file__).parent / "config.py", "r") as conf:
                out.write(conf.read())

        return