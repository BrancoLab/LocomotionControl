import sys

sys.path.append("./")

import numpy as np
from PIL import Image
from skimage.morphology import medial_axis, skeletonize
from skimage.util import invert
from scipy import signal
from typing import Tuple

from geometry import Path, Vector
from geometry import interpolate
from kinematics import track_cordinates_system as TCS
from data.data_utils import convolve_with_gaussian


"""
    Extracts a 'center' line representing a track centered through the arena
    and uses an algoirthm from:
        http://phungdinhthang.com/2016/12/16/calculate-racing-lines-automatically/?i=1

    to compute best traces that minimizes length or angle.
"""


def kernel_gaussian_2D(size: int = 5):
    # define normalized 2D gaussian
    def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
        return (
            1.0
            / (2.0 * np.pi * sx * sy)
            * np.exp(
                -(
                    (x - mx) ** 2.0 / (2.0 * sx ** 2.0)
                    + (y - my) ** 2.0 / (2.0 * sy ** 2.0)
                )
            )
        )

    x = np.linspace(-2, 2, num=size)
    y = np.linspace(-2, 2, num=size)
    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D
    z = gaus2d(x, y)
    return z


def get_skeleton() -> Tuple[np.ndarray, np.ndarray]:
    """
        It skeletonizes an image of the hairpin
    """
    # load arena image & threshold
    img = Image.open(
        "/Users/federicoclaudi/Documents/Github/LocomotionControl/draw/hairpin.png"
    )
    new_width = 40
    new_height = 60
    img = img.resize((new_width, new_height))
    img = np.array(img)[:, :, 0]
    img[img > 0] = 1
    arena = invert(img)
    arena[arena == 254] = 0
    arena[arena == 255] = 1

    # perform skeletonization
    skeleton = skeletonize(arena)
    _, distance = medial_axis(arena, return_distance=True)

    # convolve distance map with gaussian kernel
    kernel = kernel_gaussian_2D()
    distance = signal.convolve2d(
        distance, kernel, boundary="symm", mode="same"
    )

    return skeleton, distance


def skeleton2path(skeleton: np.ndarray, points_spacing: int) -> Path:
    """
        It finds points on the skeleton and sorts
        them absed on their position.
        Returns a Path over the skeleton (smoothed)
    """
    y, x = np.where(skeleton)
    points = np.array([x, y]).T
    p = points[0]
    pts = [p]
    added = [0]
    for n in range(len(points)):
        # get the closest point to the current one
        dists = np.apply_along_axis(np.linalg.norm, 1, points - p)

        for idx in added:
            dists[idx] = 10000

        p = points[np.argmin(dists)]
        pts.append(p)
        added.append(np.argmin(dists))

    pts = pts[3:-3]
    X = convolve_with_gaussian(np.float64([p[0] + 1 for p in pts])[::-1])
    Y = convolve_with_gaussian(np.float64([p[1] + 1 for p in pts])[::-1])

    return Path(X, Y).downsample(spacing=points_spacing)


def compute_extruded_paths(
    path: Path, width_at_points: np.ndarray
) -> Tuple[Path, Path]:
    """
        Computes paths extruded to left and right of original path based on widht
        of path at each point
    """
    L, R = dict(x=[], y=[]), dict(x=[], y=[])  # left, right offset paths
    for n in range(len(path)):
        theta = np.radians(path.normal.angle[n])
        p = path[n]
        width = width_at_points[n]

        L["x"].append(p.x + width * np.cos(theta))
        L["y"].append(p.y + width * np.sin(theta))

        R["x"].append(p.x - width * np.cos(theta))
        R["y"].append(p.y - width * np.sin(theta))

    left_line = Path(L["x"], L["y"])
    right_line = Path(R["x"], R["y"])

    return left_line, right_line


def extract_track_from_image(
    points_spacing: int = 3,  # distance in cm between points along the track
    k: int = 6,  # number of control points for each track point
) -> Tuple[Path, dict]:
    """
        It loads an image with Hairpin, uses image processing to extract
        a skeleton of the track (midpoint between side walls) and smooth it.
        The distance along the skeleton from the wall is used to define the widht of the track.
        Thus the 'track' is a line through the hairpin + a width at each point.

        It then define, for each point along the track, K many 'control points' along the 
        normal direction on either side of the track.
    """
    skeleton, distance = get_skeleton()

    # create path by finding points along the skeleton and smoothing the result
    center_line = skeleton2path(skeleton, points_spacing)
    l = len(center_line)

    # get the arena width at each point
    track_width = [
        distance[int(center_line[n].y), int(center_line[n].x)]
        for n in range(l)
    ]
    width_at_points = [
        track_width[n] + (1 if n > l * 0.5 else 0.5) for n in range(l)
    ]

    # get left/right side paths
    left_line, right_line = compute_extruded_paths(
        center_line, width_at_points
    )

    # ? compute control points
    # for each line path, create N control points orthogonal to the path
    P = np.linspace(0, 1, k)
    control_points = {}
    for n in range(len(center_line)):
        l = left_line[n]
        r = right_line[n]

        # create control points as Vector
        x = [interpolate.linear(l.x, r.x, p) for p in P]
        y = [interpolate.linear(l.y, r.y, p) for p in P]

        control_points[n] = Vector(x, y)

    # get center and side tracks projected to the center's coordinates system
    left_to_track = TCS.path_to_track_coordinates_system(
        center_line, left_line
    )
    center_to_track = TCS.path_to_track_coordinates_system(
        center_line, center_line
    )
    right_to_track = TCS.path_to_track_coordinates_system(
        center_line, right_line
    )

    return (
        left_line,
        center_line,
        right_line,
        left_to_track,
        center_to_track,
        right_to_track,
        control_points,
    )


# ------------------------------- fit best line ------------------------------ #


class NodeValue:
    def __init__(self, k):
        self.value_coming_from = np.zeros(k)  # value coming from previous
        self.next_best_node = np.zeros(
            k
        )  # index of next best coming from previous


def compute_nodes_values(control_points: dict, k: int, **kwargs) -> dict:
    """
        Given a cost function, it computes the values of each control node based
        on previous/next nodes.

        See: http://phungdinhthang.com/2016/12/16/calculate-racing-lines-automatically/?i=1
    """

    def segment_cost(
        prev: Vector,
        curr: Vector,
        next: Vector,
        angle_cost: float = 0,
        length_cost: float = 1,
    ):
        """
            Cost at one node based on previous and nesxt nodes
        """
        A, B = (curr - prev), (next - curr)
        # compute angle between previous and next segment
        theta = np.radians(A.angle - B.angle)

        # compute arc length
        L = A.magnitude + B.magnitude

        return angle_cost * np.cos(theta) - length_cost * L

    node_values = {
        n: [NodeValue(k) for i in range(k)] for n in control_points.keys()
    }

    for line in range(len(control_points) - 1)[1:][::-1]:
        line_prev, line_curr, line_next = line - 1, line, line + 1

        for c, curr in enumerate(control_points[line_curr]):
            for p, prev in enumerate(control_points[line_prev]):
                best_value = -10000
                best_idx = 0

                for n, nxt in enumerate(control_points[line_next]):
                    # compute the value of curr, coming from prev and going to next
                    value = segment_cost(prev, curr, nxt, **kwargs)

                    # get the value of next coming from curr
                    value += node_values[line_next][n].value_coming_from[c]

                    # keep best
                    if value > best_value:
                        best_value = value
                        best_idx = n

                # update value of coming to current from previous and next best from current coming from previous
                node_values[line_curr][c].value_coming_from[p] = best_value
                node_values[line_curr][c].next_best_node[p] = int(best_idx)

    return node_values


def fit_best_trace(
    control_points: dict,
    center_line: Path,
    k: int,
    angle_cost: float = 1,
    length_cost: float = 1,
) -> Tuple[Path, Path]:
    """
        Fits a best trace through the arena based on the cost for length and angle.
        It returns a Path with the best trace and one with the best trace in the track's coordinates
        system
    """
    node_values = compute_nodes_values(
        control_points, k, angle_cost=angle_cost, length_cost=length_cost
    )

    n = 0
    prev = 0
    node = 0
    trace = [node]
    while n < len(node_values) - 2:
        # get the next best node, at this node and coming from the previous
        next_best = node_values[n][node].next_best_node[prev]
        prev = node
        node = int(next_best)
        trace.append(node)

        n += 1

    trace_path = Path(
        [control_points[n][v].x for n, v in enumerate(trace)],
        [control_points[n][v].y for n, v in enumerate(trace)],
    )
    trace_to_track = TCS.path_to_track_coordinates_system(
        center_line, trace_path
    )

    return trace_path, trace_to_track


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from myterial import blue_grey
    import draw

    (
        left_line,
        center_line,
        right_line,
        left_to_track,
        center_to_track,
        right_to_track,
        control_points,
    ) = extract_track_from_image(points_spacing=1, k=9)

    min_len_trace_path, min_len_trace_to_track = fit_best_trace(
        control_points, center_line, 9, angle_cost=0, length_cost=1
    )
    min_ang_trace_path, min_ang_trace_to_track = fit_best_trace(
        control_points, center_line, 9, angle_cost=1, length_cost=0
    )

    # ? plot
    f = plt.figure(figsize=(8, 12))
    axes = f.subplot_mosaic(
        """
            AAA
            AAA
            AAA
            BBB
        """
    )

    # draw tracking 2d
    draw.Tracking(center_line.x, center_line.y, ax=axes["A"])
    draw.Tracking.scatter(
        center_line.x, center_line.y, ax=axes["A"], color=blue_grey
    )
    draw.Tracking(left_line.x, left_line.y, ls="--", alpha=0.5, ax=axes["A"])
    draw.Tracking(
        right_line.x, right_line.y, ls="--", alpha=0.5, color="b", ax=axes["A"]
    )

    for pts in control_points.values():
        draw.Tracking.scatter(
            pts.x, pts.y, color=[0.7, 0.7, 0.7], ax=axes["A"], s=25, alpha=0.5
        )

    _ = draw.Hairpin(alpha=1, ax=axes["A"])

    # track tracking in track coord system
    draw.Tracking(
        left_to_track.x, left_to_track.y, ax=axes["B"], ls="--", color="k"
    )
    draw.Tracking(
        center_to_track.x, center_to_track.y, ax=axes["B"], lw=2, color="k"
    )
    draw.Tracking(
        right_to_track.x, right_to_track.y, ax=axes["B"], ls="--", color="b"
    )

    # draw best trace
    draw.Tracking(
        min_len_trace_path.x,
        min_len_trace_path.y,
        color="salmon",
        ax=axes["A"],
        lw=3,
    )
    draw.Tracking(
        min_len_trace_to_track.x,
        min_len_trace_to_track.y,
        color="salmon",
        ax=axes["B"],
    )

    draw.Tracking(
        min_ang_trace_path.x,
        min_ang_trace_path.y,
        color="green",
        ax=axes["A"],
        lw=3,
    )
    draw.Tracking(
        min_ang_trace_to_track.x,
        min_ang_trace_to_track.y,
        color="green",
        ax=axes["B"],
    )

    _ = axes["B"].set(ylim=[-5, 5])

    plt.show()
