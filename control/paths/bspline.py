import sys

sys.path.append("./")

import numpy as np
import scipy.interpolate as scipy_interpolate

from control_fc.paths.waypoints import Waypoints
from geometry import Path

"""
    Code adapted from: https://github.com/zhm-real/CurvesGenerator
    zhm-real shared code to create different types of paths under an MIT license.
"""


def BSpline(
    x: np.ndarray,
    y: np.ndarray,
    n_path_points: int = 500,
    degree=3,
    cut: float = 0.06,
    fps: int = 60,
) -> Path:
    ipl_t = np.linspace(0.0, len(x) - 1, len(x))
    spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree)
    spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree)

    travel = np.linspace(0.0, len(x) - 1, n_path_points)

    if cut:
        cut = int(cut * n_path_points)
        return Path(
            spl_i_x(travel)[cut:-cut], spl_i_y(travel)[cut:-cut], fps=fps
        )
    else:
        return Path(spl_i_x(travel), spl_i_y(travel), fps=fps)


if __name__ == "__main__":

    import sys

    sys.path.append("./")

    import matplotlib.pyplot as plt
    import pandas as pd
    from myterial import pink

    import draw

    f, ax = plt.subplots(figsize=(7, 10))

    # load and draw tracking data
    from fcutils.path import files

    for fp in files(
        "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/control/",
        # r'D:\Dropbox (UCL)\Rotation_vte\Locomotion\analysis\control',
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
        draw.Arrow(
            wp.x, wp.y, wp.theta, 2.5, width=4, color=pink, outline=True,
        )

    # fit splines
    spline = BSpline(wps.x, wps.y, degree=3)
    draw.Tracking(spline.x, spline.y, lw=4, color="k", label="spline")

    ax.legend()

    from fcutils.plot.figure import save_figure

    save_figure(
        f,
        "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Presentations/Presentations/Fiete lab/tracking_directoins",
        svg=True,
    )

    plt.show()
