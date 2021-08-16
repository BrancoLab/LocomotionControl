import numpy as np
from loguru import logger
import pandas as pd

from data.dbase.io import load_bin


def load_session_data(table, session):
    """
        loads and cleans up the bonsai data for one session
    """
    logger.debug(
        f'Loading Bonsai Behavior data for session: {session["name"]}'
    )

    # load analog
    analog = load_bin(
        session["ai_file_path"], nsigs=session["n_analog_channels"]
    )

    # get analog inputs between frames start/end times
    _analog = (
        analog[session["bonsai_cut_start"] : session["bonsai_cut_end"]] / 5
    )
    session["pump"] = 5 - _analog[:, 1]  # 5 -  to invert signal
    session["speaker"] = _analog[:, 2]

    # load csv data
    logger.debug("Loading CSV")
    data = pd.read_csv(session["csv_file_path"])
    if len(data.columns) < 5:
        logger.warning("Skipping because of incomplete CSV")
        return None  # first couple recordings didn't save all data

    data.columns = [
        "ROI activity",
        "lick ROI activity",
        "mouse in ROI",
        "mouse in lick ROI",
        "deliver reward signal",
        "reward available signal",
    ]

    # make sure csv data has same length as the number of frames (off by max 2)
    delta = session["n_frames"] - len(data)
    if delta > 2:
        raise ValueError(
            f"We got {session['n_frames']} frames but CSV data has {len(data)} rows"
        )
    if not delta:
        raise NotImplementedError("This case is not covered")
    pad = np.zeros(delta)

    # add key entries
    session["reward_signal"] = np.concatenate(
        [data["deliver reward signal"].values, pad]
    )
    session["trigger_roi"] = np.concatenate([data["mouse in ROI"].values, pad])
    session["reward_roi"] = np.concatenate(
        [data["mouse in lick ROI"].values, pad]
    )
    session["reward_available_signal"] = np.concatenate(
        [data["reward available signal"].values, pad]
    )
    return session
