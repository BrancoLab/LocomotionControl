import sys

sys.path.append("./")

import numpy as np
import scipy.interpolate as scipy_interpolate
from collections import namedtuple

from control.waypoints import Waypoints

"""
    Code adapted from: https://github.com/zhm-real/CurvesGenerator
    zhm-real shared code to create different types of paths under an MIT license.
"""

path = namedtuple("path", "x, y")


def interpolate_b_spline_path(
    x: np.ndarray,
    y: np.ndarray,
    n_path_points: int = 500,
    degree=3,
    cut: float = 0.06,
):
    ipl_t = np.linspace(0.0, len(x) - 1, len(x))
    spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree)
    spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree)

    travel = np.linspace(0.0, len(x) - 1, n_path_points)

    cut = int(cut * n_path_points)
    return path(spl_i_x(travel)[cut:-cut], spl_i_y(travel)[cut:-cut])


if __name__ == "__main__":

    import sys

    sys.path.append("./")

    import matplotlib.pyplot as plt
    import pandas as pd

    import draw

    from control.dubins_path import DubinPath

    f, ax = plt.subplots(figsize=(7, 10))

    # load and draw tracking data
    from fcutils.path import files

    for fp in files(
        "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/control/",
        "*.h5",
    ):
        tracking = pd.read_hdf(fp, key="hdf")
        tracking.x = 20 - tracking.x + 20
        draw.Tracking(tracking.x, tracking.y, alpha=0.7)

    # draw hairpin arena
    draw.Hairpin(ax)

    # draw waypoints
    wps = Waypoints(use="spline")
    for wp in wps:
        draw.Arrow(wp.x, wp.y, wp.theta, 2, width=4, color="g")

    # fit splines
    spline = interpolate_b_spline_path(wps.x, wps.y, degree=3)
    draw.Tracking(spline.x, spline.y, lw=4, color="k", label="spline")

    # fit and draw dubin path
    dubin = DubinPath(Waypoints()).fit()
    draw.Tracking(dubin.x, dubin.y, color="r", label="dubin")

    ax.legend()
    plt.show()
