"""
Converts probes reconstructed in 3D in the samples spaces to coordinates in the atlas space
using brainreg's deformation fields.
"""

import pandas as pd
import numpy as np
import tifffile
from pathlib import Path
from brainrender import Scene
from brainrender.actors import Points

# from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev

from bg_atlasapi.bg_atlas import BrainGlobeAtlas

atlas = BrainGlobeAtlas("allen_mouse_25um")


source = Path(r"W:\swc\branco\BrainSaw\YI_0000012b\brainreg")

track_files = list(
    (source / "manual_segmentation" / "sample_space" / "tracks").glob(
        "*.points"
    )
)
deformation_fields_files = list(source.glob("deformation_field_*.tiff"))
deformation_fields = [tifffile.imread(df) for df in deformation_fields_files]


def transform_points_downsampled_to_atlas_space(
    downsampled_points, atlas, deformation_fields
):
    field_scales = [int(1000 / resolution) for resolution in atlas.resolution]
    points = [[], [], []]
    for axis, deformation_field in enumerate(deformation_fields):

        for point in downsampled_points:
            point = [int(round(float(p))) for p in point]
            points[axis].append(
                int(
                    round(
                        field_scales[axis]
                        * deformation_field[point[0], point[1], point[2]]
                    )
                )
            )

    transformed_points = np.array(points).T
    return transformed_points


def spline_fit(points, smoothing=0.2, k=3, n_points=100):
    """Given an input set of 2/3D points, returns a new set of points
    representing the spline interpolation
    Parameters
    ----------
    points : np.ndarray
        2/3D array of points defining a path
    smoothing : float
        Smoothing factor
    k : int
        Spline degree
    n_points : int
        How many points used to define the resulting interpolated path
    Returns
    ----------
    new_points : np.ndarray
        Points defining the interpolation
    """

    # scale smoothing to the spread of the points
    max_range = max(np.max(points, axis=0) - np.min(points, axis=0))
    smoothing *= max_range

    # calculate bspline representation
    tck, _ = splprep(points.T, s=smoothing, k=k)

    # evaluate bspline
    spline_fit_points = splev(np.linspace(0, 1, n_points), tck)

    return np.array(spline_fit_points).T


# def upsample_points(points, track_length):
#     x = points[:, 0]
#     y = points[:, 1]
#     z = points[:, 2]

#     t = np.arange(len(x))
#     qs_x = CubicSpline(t, x)
#     qs_y = CubicSpline(t, y)
#     qs_z = CubicSpline(t, z)

#     upsampled_t = np.linspace(0, len(x), int(track_length))
#     upsampled_x = qs_x(upsampled_t)
#     upsampled_y = qs_y(upsampled_t)
#     upsampled_z = qs_z(upsampled_t)

#     return np.vstack((upsampled_x, upsampled_y, upsampled_z)).T


scene = Scene()
scene.add_brain_region("MOs", alpha=0.3)
for track_file in track_files:
    track_df = pd.read_hdf(track_file).values
    track_length = np.sqrt(sum((track_df[-1] - track_df[0]) ** 2)) * 25

    track_points = transform_points_downsampled_to_atlas_space(
        track_df, atlas, deformation_fields
    )
    track = spline_fit(track_points, n_points=int(track_length))

    scene.add(Points(track_points * 25, colors="red", radius=30))
    scene.add(Points(track * 25, colors="black"))

    # save track to file
    np.save(track_file.with_suffix(".npy"), track)

scene.render()
