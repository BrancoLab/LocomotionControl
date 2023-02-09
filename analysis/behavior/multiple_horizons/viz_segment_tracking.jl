using jcontrol
import jcontrol.comparisons: track_segments
using jcontrol.visuals
import jcontrol: trimtrial
using Plots

"""
Load tracking data and cut between start and end of selected
segment, then visualize tracking and other variables.
"""

segment = track_segments[3]

# load & trim trials
trials = @time load_cached_trials(; keep_n=20)
trials = map(t -> trimtrial(t, segment), trials)

varvalues = Dict()
for var in (:x, :y, :θ, (:u):ω)
    varvalues[var] = Dict(
        "start" => get_varvalue_at_frameidx(trials, var, 1),
        "stop" => get_varvalue_at_frameidx(trials, var),
    )
end

# TODO draw stuff
# ------------------------------- visualization ------------------------------ #
# draw tracking data
p1 = draw(:arena)
draw!(track; alpha=0.1)
draw!(segment)
draw!.(trials)

function plotdist(key)
    hist = histogram(varvalues[key]["start"]; label=string(key) * " init")
    hist = histogram!(varvalues[key]["stop"]; label=string(key) * " final")
    return hist
end

plots = map(plotdist, collect(keys(varvalues)))

l = @layout [
    a{0.5w} grid(length(keys(varvalues)), 1)
]

closeall()
display(plot(p1, plots...; layout=l))
