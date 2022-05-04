import numpy as np
import json


class Track:
    def __init__(self):
        # load track from json
        with open("analysis/behavior/rl/track.json", "r") as fin:
            track = json.load(fin)

        self.x = np.array(track["X"])
        self.y = np.array(track["Y"])
        self.theta = np.array(track["θ"])
        self._curv = np.array(track["curvature"])
        self.S = np.array(track["S"])
        self.width = np.array(track["width"])


    def get_at_sval(self, s):
        """
            Get the x,y,theta at a given s value
        """
        idx = np.argmin(np.abs(self.S - s))
        return self.x[idx], self.y[idx], self.theta[idx]

    def curvature(self, s):
        """
            Get the curvature at a given s value
        """
        idx = np.argmin(np.abs(self.S - s))
        return self._curv[idx]

    def w(self, s):
        """
            Return width at given s value
        """
        idx = np.argmin(np.abs(self.S - s))
        return self.width[idx]

    def s(self, x, y):
        """
            Get the s value of the track's closest point to (x,y)
        """
        # get closest point
        idx = np.argmin(np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2))
        return self.S[idx]
