# import pandas as pd
from typing import List
from loguru import logger
import sys
import numpy as np

sys.path.append("./")

from fcutils.progress import track
from fcutils.path import files, from_json

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
    bouts_files = files(
        paths.analysis_folder / "behavior" / "saved_data",
        "*_complete_bout.json",
    )

    # load from json and sort based on duration
    bouts_json = []
    for n, bf in track(
        enumerate(bouts_files),
        total=len(bouts_files),
        description="loading bouts JSON",
        transient=True,
    ):
        bj = from_json(bf)
        if bj["duration"] > duration_max or bj["gcoord"][0] > 0.1:
            continue
        else:
            bouts_json.append(bj)
    durations = [bj["duration"] for bj in bouts_json]
    bouts_json = [bouts_json[idx] for idx in np.argsort(durations)][:keep]
    logger.debug(
        f"Loaded and kept {len(bouts_json)} JSON files for locomotor bouts"
    )

    for bout in track(bouts_json, description="loading bouts", transient=True):
        bouts.append(
            LocomotionBout(
                bout,
                linearize_to=linearize_to,
                window=window,
                trim=trim,
                bparts_tracking=bout,
            )
        )

    return bouts


if __name__ == "__main__":
    bouts = load_complete_bouts(keep=2)
    print(bouts[0])
