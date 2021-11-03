import numpy as np
from typing import Tuple

import fcutils.progress

from geometry import Path, Vector

"""
    Code to project tracking data to a coordinates system along
    a path
"""


def point_to_track_coordinates_system(
    track: Path, point: Vector
) -> Tuple[float, float]:
    """
        Given a point it finds the closest point on a Path and it returns the 
        distance of it along the track and the distance of the point along the track
    """
    # get closest track point
    dists = np.apply_along_axis(
        np.linalg.norm, 1, track.points - point.as_array()
    )
    track_point_idx = np.argmin(dists)

    # get distance along the track until that point
    path_length = np.sum(track.speed[:track_point_idx]) / track.fps

    # get distance along normal direction
    track_point = Vector(track.x[track_point_idx], track.y[track_point_idx])
    dist_vec = point - track_point
    normal_distance = dist_vec.dot(track.normal[track_point_idx])

    return path_length, normal_distance


def path_to_track_coordinates_system(track: Path, path: Path, progress:bool=False) -> Path:
    """
        Given a Path representing e.g. the shortest path through the arena
        and another Path with e.g. tracking from a mouse, this function returns
        the second path in a coordinate frame centerd on the first (track).
        The X axis of this frame denotes distance along the track, while the second
        the distance away (orthogonal) to the track
    """
    X, Y = [], []
    def empty_track(*args):
        return args

    if progress:
        track = fcutils.progress.track
    else:
        track = empty_track

    for n in track(range(len(path))):
        x, y = point_to_track_coordinates_system(track, path[n])
        X.append(x)
        Y.append(y)

    return Path(np.array(X), np.array(Y))
