import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("./")

from myterial import green

import draw
import control


# load tracking
bouts = pd.read_hdf(
    "/Users/federicoclaudi/Dropbox (UCL)/Rotation_vte/Locomotion/analysis/behavior/roi_crossings/T2_crossings.h5"
)
bout = bouts.iloc[20]

# get waypoints
wps = control.paths.Waypoints.from_list(
    [
        control.paths.Waypoint(x=10, y=25, theta=100, speed=64,),
        control.paths.Waypoint(x=10, y=25, theta=100, speed=64,),
        control.paths.Waypoint(x=11, y=40, theta=75, speed=64,),
        control.paths.Waypoint(x=20.0, y=47, theta=0, speed=49,),
        control.paths.Waypoint(x=28.0, y=40, theta=280, speed=49,),
        control.paths.Waypoint(x=29, y=28, theta=270, speed=55,),
        control.paths.Waypoint(x=29, y=28, theta=270, speed=55,),
    ]
)

# fit bspline
spline = control.paths.interpolate_b_spline_path(wps.x, wps.y, cut=0.2)

# draw stuff
f, ax = plt.subplots(figsize=(5, 5))

# draw arena and tracking
draw.T2(set_ax=True, shade=True)
draw.Tracking(bouts.x, bouts.y, alpha=0.5)
draw.Tracking(bout.x, bout.y, color="k")

# draw waypoints
draw.Arrows(wps.x, wps.y, wps.theta, L=2, color="salmon")

# draw spline
draw.Tracking(spline.x, spline.y, lw=3, color=green, alpha=0.75)

plt.show()
