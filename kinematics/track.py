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
    Code to reconstruct a track made of a center line and two side
    lines of varying width.
    This track can then be used to project the tracking data to
    its reference frame.
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

    pts = pts[2:-2]
    X = np.float64([p[0] + 1 for p in pts])[::-1]
    Y = np.float64([p[1] + 1 for p in pts])[::-1] - 0.5

    # adjust coordinates to make it look more like the animals tracking (e.g. because mice have non-zero width)
    Y[Y < 10] -= 2.5
    Y[Y > 50] += 1.5
    Y[(Y > 40) & (Y < 50) & (5 < X) & (X < 33)] += 2
    X[(X > 8) & (X < 20) & (Y > 40) & (Y < 50)] -= 2
    X[(X > 20) & (X < 33) & (Y > 40) & (Y < 50)] += 2

    Y[(X > 10) & (X < 15) & (Y > 43) & (Y < 48)] += 2.5

    X[(Y < 18) & (X > 8) & (X < 14)] -= 1
    X[(Y < 18) & (X > 24) & (X < 32)] -= 1
    X[(Y < 18) & (X > 16) & (X < 25)] += 1
    X[(Y < 18) & (X > 32)] += 1

    Y[(X > 30) & (X < 35) & (Y < 5)] += 1

    # smooth
    X = convolve_with_gaussian(X, kernel_width=31)
    Y = convolve_with_gaussian(Y, kernel_width=31)

    return Path(X, Y).downsample(spacing=points_spacing)


def compute_extruded_paths(
    path: Path, width_at_points: np.ndarray
) -> Tuple[Path, Path]:
    """
        Computes paths extruded to left and right of original path based on widht
        of path at each point
    """
    N = len(path)
    L, R = dict(x=[], y=[]), dict(x=[], y=[])  # left, right offset paths
    for n in range(N):
        theta = np.radians(path.normal.angle[n])
        p = path[n]
        width = width_at_points[n]

        # hardcoded width for certain segments of the arena
        if n > N * 0.6 and n < N * 0.85:
            width = 3
        elif 0.2 * N < n < 0.45 * N:
            width = 3
        elif n > N * 0.90 or n < N * 0.025:
            width = 0.5
        elif n > N * 0.85:
            width = 1.5
        else:
            width = 2

        L["x"].append(p.x + width * np.cos(theta))
        L["y"].append(p.y + width * np.sin(theta))

        R["x"].append(p.x - width * np.cos(theta))
        R["y"].append(p.y - width * np.sin(theta))

    left_line = Path(L["x"], L["y"])
    right_line = Path(R["x"], R["y"])

    return left_line, right_line


def extract_track_from_image(
    points_spacing: int = 3,  # distance in cm between points along the track
    k: int = 5,  # number of control points for each track point
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
        track_width[n] + (0.25 if n > l * 0.5 else 0) for n in range(l)
    ]
    # width_at_points = [
    #     track_width[n] for n in range(l)
    # ]

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
