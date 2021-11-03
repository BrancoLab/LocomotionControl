import numpy as np
from typing import List

from geometry import Path


'''
    Functions to do time-rescaling on a set of paths
'''


def time_rescale(paths:List[Path]) -> List[Path]:
    '''
        Rescales all paths in a list to have the same
        number of frames to the one with the smallest number of frames
        and sets time to go in 0-1 range in that interval.
    '''

    n_frames = np.min([len(path) for path in paths])
    resampled_paths = [p.downsample_in_time(n_frames) for p in paths]
    return resampled_paths


def average_xy_trajectory(paths:List[Path], rescale:bool=False) -> Path:
    '''
        Computes the average XY trajectory from a set of paths,
        rescaling them in time if necessary
    '''
    if rescale:
        paths = time_rescale(paths)

    X = np.mean(np.vstack([path.x for path in paths]), 0)
    Y = np.mean(np.vstack([path.y for path in paths]), 0)

    return Path(X, Y)
