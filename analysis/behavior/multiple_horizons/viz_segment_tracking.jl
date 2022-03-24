using jcontrol
import jcontrol.comparisons: track_segments

"""
Load tracking data and cut between start and end of selected
segment, then visualize tracking and other variables.
"""

segment = track_segments[1]



# load data
trials = @time load_trials(; keep_n=20)
trials = @time map(t->Trial(t, FULLTRACK), eachrow(trials))



# # viz
# p1 = draw(:arena)
# draw!(track; alpha=.1)
# draw!(segment)

# display(p1)