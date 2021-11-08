import pandas as pd
from typing import List
from loguru import logger

from fcutils.progress import track

from data.data_structures import LocomotionBout
from data import paths
from geometry import Path


def load_complete_bouts(
    duration_max: float = 10,
    keep: int = -1,
    linearize_to: Path = None,
    window: int = 4,
    trim: bool = True,
) -> List[LocomotionBout]:
    """
        Loads complete bouts saved as a .h5 file
    """
    # load and clean complete bouts
    _bouts = pd.read_hdf(
        paths.analysis_folder
        / "behavior"
        / "saved_data"
        / f"complete_bouts.h5"
    )
    _bouts = _bouts.loc[
        (_bouts.start_roi == 0) & (_bouts.duration < duration_max)
    ]
    _bouts = _bouts.sort_values("duration").iloc[:keep]
    logger.debug(f"Kept {len(_bouts)} bouts")

    # turn into Locomotion bouts
    bouts = []
    for i, bout in track(
        _bouts.iterrows(), total=len(_bouts), description="loading bouts..."
    ):
        bouts.append(
            LocomotionBout(
                bout, linearize_to=linearize_to, window=window, trim=trim
            )
        )

    return bouts
