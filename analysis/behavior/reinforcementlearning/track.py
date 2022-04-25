import numpy as np
import json


class Track:
    def __init__(self):
        # load track from json
        with open("track.json", "r") as fin:
            track = json.read(fin)

        self.x = track["X"]
        self.y = track["Y"]
        self._curv = track["curvature"]

    def s(self, x, y):
        """
            Get the s value of the track's closest point to (x,y)
        """
        # get closest point
        idx = np.argmin(np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2))
        return self._curv[idx]
