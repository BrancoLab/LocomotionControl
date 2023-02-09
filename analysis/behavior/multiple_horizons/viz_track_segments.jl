using jcontrol.visuals
using jcontrol
import jcontrol.comparisons: track_segments

using Plots
import MyterialColors: teal, indigo_dark

"""
Visualize all the track's segments used for the multiple
horizons analysis
"""

track = FULLTRACK

p1 = draw(:arena)
draw!(track; alpha=0.1)
draw!(track_segments[1])

p2 = draw(:arena)
draw!(track; alpha=0.1)
draw!(track_segments[2])

p3 = draw(:arena)
draw!(track; alpha=0.1)
draw!(track_segments[3])

p4 = draw(:arena)
draw!(track; alpha=0.1)
draw!(track_segments[4])

p5 = draw(:arena)
draw!(track; alpha=0.1)
draw!(track_segments[5])

p6 = draw(:arena)
draw!(track; alpha=0.1)
draw!(track_segments[6])

display(plot(p1, p2, p3, p4, p5, p6; layout=grid(2, 3), size=(1200, 1200)))
