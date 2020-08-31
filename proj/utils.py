import numpy as np

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
    return ( angles + np.pi) % (2 * np.pi ) - np.pi