import pandas as pd
from typing import List
import numpy as np
from loguru import logger
import sys

sys.path.append("./")

from fcutils.progress import track

from kino.locomotion import Locomotion
from kino.animal import mouse

from data import paths
from geometry import Path


def load_complete_bouts(
    duration_max: float = 10,
    keep: int = -1,
    linearize_to: Path = None,
    window: int = 4,
    trim: bool = True,
) -> List[Locomotion]:
    """
        Loads complete bouts saved as a .h5 file
    """
    # load and clean complete bouts
    _bouts = pd.read_hdf(
        paths.analysis_folder
        / "behavior"
        / "saved_data"
        / "complete_bouts.h5",
        key="hdf",
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
        locomotion_bout = Locomotion(mouse, bout, fps=60,)

        # add extra stuff from the locomotion bout (e.g. gcoord)
        for key in bout.index:
            if "_x" not in key and "_y" not in key:
                if isinstance(bout[key], list):
                    bout_data = np.array(bout[key])
                else:
                    bout_data = bout[key]
                setattr(locomotion_bout, key, bout_data)

        bouts.append(locomotion_bout)

    return bouts
