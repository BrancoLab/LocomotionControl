import numpy as np
from dataclasses import dataclass
from math import sqrt
import matplotlib.pyplot as plt

from myterial import (
    salmon_dark,
    orange,
    amber,
    green,
    teal,
    cyan_dark,
    blue,
    indigo,
    purple,
    pink,
    blue_grey,
    brown,
    black,
)
from myterial.utils import map_color


@dataclass
class Segment:
    p0: tuple  # start point
    p1: tuple  # end point
    iid: int  # ID number
    name: str  # name
    color: str  # color
    last: bool = False

    @property
    def length(self):
        return sqrt(
            (self.p0[0] - self.p1[0]) ** 2 + (self.p0[1] - self.p1[1]) ** 2
        )

    def interpolate(self, n_points: int):
        """
            Creates a line segment between the two points
        """
        p0, p1 = np.array(self.p0), np.array(self.p1)
        line = []
        # interpolate between knots
        for step in np.linspace(0, 1, n_points):
            line.append(p0 * (1 - step) + p1 * step)
        self.line = np.vstack(line)
        self.ids = (
            np.ones(len(self.line)) * self.iid
        )  # assign at each point the segment's ID
        return self.line

    def draw(self, ax: plt.axis):
        ax.scatter(
            *self.p0,
            s=200,
            lw=1,
            ec=[0.3, 0.3, 0.3],
            color=self.color,
            zorder=100,
        )
        if self.last:
            ax.scatter(
                *self.p1,
                s=200,
                lw=1,
                ec=[0.3, 0.3, 0.3],
                color=self.color,
                zorder=100,
            )

        ax.plot(
            [self.p0[0], self.p1[0]],
            [self.p0[1], self.p1[1]],
            lw=6,
            color=[0.3, 0.3, 0.3],
            zorder=98,
        )
        ax.plot(
            [self.p0[0], self.p1[0]],
            [self.p0[1], self.p1[1]],
            lw=5,
            color=self.color,
            zorder=99,
        )


POINTS = [  # hand defined points along the track
    (20, 40),  # start
    (22, 8),  # end of first corridor
    (12, 8),  # first bend
    (12, 45),  # end of second corridor
    (28, 45),  # end of second bend
    (28, 8),  # end of third corridor
    (38, 8),  # end of third bend
    (38, 47),  # end of fourth corridor
    (28, 58),  # end of fourth bend
    (12, 58),  # end of fifth corridor
    (2, 47),  # end of fifth bend
    (2, 2),  # goal location
    (20, 35),  # end of reward trigger area
    (2, 7),  # start of goal location
]


class HairpinTrace:
    _n_samples: int = 500  # target number of samples in trace
    trace = None  # to be filed in

    segments = [
        Segment(POINTS[0], POINTS[12], 0, "start", brown),
        Segment(POINTS[12], POINTS[1], 1, "c0", salmon_dark),
        Segment(POINTS[1], POINTS[2], 2, "b0", orange),
        Segment(POINTS[2], POINTS[3], 3, "c1", amber),
        Segment(POINTS[3], POINTS[4], 4, "b1", green),
        Segment(POINTS[4], POINTS[5], 5, "c2", teal),
        Segment(POINTS[5], POINTS[6], 6, "b2", cyan_dark),
        Segment(POINTS[6], POINTS[7], 7, "c3", blue),
        Segment(POINTS[7], POINTS[8], 8, "b3", indigo),
        Segment(POINTS[8], POINTS[9], 9, "c4", purple),
        Segment(POINTS[9], POINTS[10], 10, "b3", pink),
        Segment(POINTS[10], POINTS[13], 11, "c5", blue_grey),
        Segment(POINTS[13], POINTS[11], 11, "goal", black, last=True),
    ]

    def __init__(self):
        self.build()

    def assign_tracking(self, x: np.ndarray, y: np.ndarray):
        """
            For a set of tracking data, it gets the closest point along the trace
            and assignes that point to it
        """
        # logger.debug(f'HairpinTrace - Fitting tracking (# samples: {len(x)}) to trace')
        if self.trace is None:
            raise ValueError("Need to build trace first!")

        def find_closest(point):
            dist = np.linalg.norm(self.trace - point, axis=1)
            return np.argmin(dist)

        # get the closest trace point to each tracking point
        xy = np.vstack([x, y]).T
        closest_points = np.apply_along_axis(find_closest, 1, xy)

        # get global coordinates 0-1 range fro reach frame
        global_coord_idxs = np.linspace(0, 1, len(self.trace_ids))
        global_coordinates = global_coord_idxs[closest_points]

        # return the arena segment index for each frame
        idxs = np.zeros_like(global_coordinates).astype(np.int32)

        idxs[(global_coordinates > 0.02) & (global_coordinates < 0.11)] = 1
        idxs[(global_coordinates > 0.11) & (global_coordinates < 0.17)] = 2
        idxs[(global_coordinates > 0.17) & (global_coordinates < 0.28)] = 3
        idxs[(global_coordinates > 0.28) & (global_coordinates < 0.36)] = 4
        idxs[(global_coordinates > 0.36) & (global_coordinates < 0.48)] = 5
        idxs[(global_coordinates > 0.48) & (global_coordinates < 0.53)] = 6
        idxs[(global_coordinates > 0.53) & (global_coordinates < 0.66)] = 7
        idxs[(global_coordinates > 0.66) & (global_coordinates < 0.72)] = 8
        idxs[(global_coordinates > 0.72) & (global_coordinates < 0.78)] = 9
        idxs[(global_coordinates > 0.78) & (global_coordinates < 0.85)] = 10
        idxs[(global_coordinates > 0.85) & (global_coordinates < 0.98)] = 11
        idxs[global_coordinates > 0.98] = 12

        return idxs, global_coordinates

    @property
    def X(self):
        return np.linspace(0, 1, self.n_segments)

    @property
    def n_segments(self):
        return len(self.segments)

    @property
    def colors(self):
        """
            Returns the color of each segment
        """
        return [segment.color for segment in self.segments]

    def colors_from_segment(self, indices: np.array):
        """
            Given a 1d array of the track segment index for each frame, 
            it returns the corresponding colors
        """
        return np.array([self.segments[idx].color for idx in indices])

    def colors_from_global_coordinates(self, gcoord: np.array):
        """
            Assign a color to each point on the global coordinates scale
        """
        return [map_color(l, name="bwr", vmin=0, vmax=1) for l in gcoord]

    def build(self):
        """
            Interpolate each segment to create a piecewise linear curve where
            the nmber of samples in each segments is matched to its length.
        """
        tot_length = np.sum([seg.length for seg in self.segments])
        samples_per_unit_length = int(np.ceil(self._n_samples / tot_length))
        samples_per_segment = [
            int(seg.length * samples_per_unit_length) for seg in self.segments
        ]

        for segment, n_samples in zip(self.segments, samples_per_segment):
            segment.interpolate(n_samples)

        # get the length of each line segment to check we have the right number of samples
        self.trace_length = np.sum(
            [len(segment.line) for segment in self.segments]
        )

        # stach all segmnts into a curve
        self.trace = np.vstack([segment.line for segment in self.segments])
        self.trace_ids = np.concatenate(
            [segment.ids for segment in self.segments]
        )

    def draw(
        self,
        ax: plt.axis = None,
        tracking: dict = None,
        colorby: str = "segment",
    ):
        ax = ax or plt.subplots(figsize=(9, 9))[1]

        # draw each segment
        for segment in self.segments:
            segment.draw(ax)

        # draw tracking data
        if tracking is not None:
            if colorby == "segment":
                c = self.colors_from_segment(tracking["segment"])
            else:
                c = self.colors_from_global_coordinates(
                    tracking["global_coord"]
                )
            ax.scatter(
                tracking["x"][::5],
                tracking["y"][::5],
                c=c[::5],
                s=50,
                alpha=0.8,
            )


if __name__ == "__main__":
    tr = HairpinTrace()

    tr.draw()

    plt.show()
