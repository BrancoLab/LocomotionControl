import numpy as np
from pathlib import Path
import pandas as pd
import pickle
from loguru import logger

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
        self.record = {}
        self.info = dict()

    def append(self, *datas):
        for data in datas:
            if not isinstance(data, dict):
                data = data._asdict()

            for k, v in data.items():
                if k not in self.record.keys():
                    self.record[k] = []
                self.record[k].append(v)

    def save(self, folder, trajectory, trial):
        logger.info(f"Saving history with entries: {self.record.keys()}")

        logger.info(
            "While saving history discarding first 60 samples to remove weird stuff"
        )
        self.record = {k: v[200:] for k, v in self.record.items()}

        # Save info
        self.info["duration"] = len(self.record["x"]) * dt
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
