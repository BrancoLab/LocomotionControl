import numpy as np
from loguru import logger
import pandas as pd

from fcutils.path import size

from data.dbase.io import load_bin


def load_session_data(session:dict, key:dict, sampling_rate:int):
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
    end_cut = session['trigger_times'][-1]+session["bonsai_cut_start"]
    _analog = (
        analog[session["bonsai_cut_start"] : end_cut] / 5
    )

    # get signals in high sampling rate
    analog_data = dict(pump=5 - _analog[:, 1],  # 5 -  to invert signal
                    speaker = _analog[:, 2])

    # go from samples to frame times
    sample_every = int(sampling_rate / 60)
    for name, sample_values in analog_data.items():
        frames_values = sample_values[::sample_every]

        if not len(frames_values) != session['n_frames']:
            raise ValueError('Wrong number of frames')
        
        # add to key
        key[name] = frames_values
        
    # load csv data
    logger.debug(f"Loading CSV file ({size(session['csv_file_path'])})")
    try:
        data = pd.read_csv(session["csv_file_path"])
    except Exception:
        logger.warning(f'Failed to open csv for {session["name"]}')
        return None

    if len(data.columns) < 5:
        logger.warning("Skipping because of incomplete CSV")
        return None  # first couple recordings didn't save all data

    logger.debug('Data loaded, cleaning it up')
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
    key["reward_signal"] = np.concatenate(
        [data["deliver reward signal"].values, pad]
    )
    key["trigger_roi"] = np.concatenate([data["mouse in ROI"].values, pad])
    key["reward_roi"] = np.concatenate([data["mouse in lick ROI"].values, pad])
    key["reward_available_signal"] = np.concatenate(
        [data["reward available signal"].values, pad]
    )
    return key

