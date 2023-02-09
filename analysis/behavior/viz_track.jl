using jcontrol
using jcontrol.visuals
import jcontrol: Track

plt = draw(:arena)

track = Track(; start_waypoint=4, keep_n_waypoints=-1)
draw!(track)

display(plt)
