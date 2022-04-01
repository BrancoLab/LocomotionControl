using Plots
using Statistics: mean
import MyterialColors: blue_grey_dark, blue_dark

using jcontrol
import jcontrol.comparisons: track_segments
using jcontrol.visuals
import jcontrol: trimtrial, euclidean, FULLTRACK, Track, trim
import jcontrol.Run: run_mtm
import jcontrol.forwardmodel: solution2state

HORIZONS_LENGTH = 180  # cm
SEGMENTID = 1

# run global solution
coptions = ControlOptions(;
    u_bounds=Bounds(10, 80),
    δ_bounds=Bounds(-90, 90, :angle),
    δ̇_bounds=Bounds(-8, 8),
    ω_bounds=Bounds(-600, 600, :angle),
    v_bounds=Bounds(-5000, 5000),
    Fu_bounds=Bounds(-5000, 5000),
)

# _, bike, _, globalsolution = run_mtm(
#     :dynamics,
#     3;
#     control_options=coptions,
#     showplots=false,
#     n_iter=5000,
# )


# load data
segment = track_segments[SEGMENTID]

# load & trim trials
trials = load_cached_trials(; keep_n=80)
trimmedtrials = map(t->trimtrial(t, segment), trials)

# get initial state
initial_state = solution2state(segment.track.S[1], globalsolution)

# trim track to define planning window
track = trim(FULLTRACK, segment.s₀, HORIZONS_LENGTH)

# get the final state
final_state = solution2state(track.S[end], globalsolution)

# fit model
_, bike, _, solution = run_mtm(
    :dynamics,
    3;
    track=track,
    icond=initial_state,
    fcond=final_state,
    control_options=coptions,
    showplots=false,
    n_iter=5000,
);


# plot
p1 = draw(:arena)
draw!(track; alpha=.1)
draw!(segment)
draw!.(trimmedtrials; lw=.75,)

plot_bike_trajectory!(globalsolution, bike; showbike=false, color=blue_grey_dark, lw=4, label=nothing)
plot_bike_trajectory!(solution, bike; showbike=false, label=nothing)

scatter!(
    [solution.x[1], solution.x[end]],
    [solution.y[1], solution.y[end]];
    ms=8,
    coor="red",
    label=nothing,
)

# draw initial and final states
draw!(initial_state)
draw!(final_state; color=blue_dark)
display(p1)


nothing
