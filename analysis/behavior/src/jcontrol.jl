module jcontrol
# using Term
# install_term_logger()


using Plots
using Interpolations

include("utils.jl")
include("kinematics.jl")
include("track.jl")
include("bike.jl")
include("control.jl")
include("forward_model.jl")
include("comparisons.jl")
include("run.jl")
include("trial.jl")
include("visuals.jl")
include("io.jl")

export Track, get_track_borders, FULLTRACK
export State, Bicycle
export ControlOptions, create_and_solve_control, Bounds
export Solution, run_forward_model
export summary_plot
export PATHS, load_trials, load_cached_trials
export arena
export DynamicsProblem, KinematicsProblem, realistict_control_options
export get_comparison_points
export run_mtm
export Trial, trimtrial, get_varvalue_at_frameidx

using .bicycle: State, Bicycle
using .control:
    ControlOptions,
    create_and_solve_control,
    Bounds,
    State,
    DynamicsProblem,
    KinematicsProblem,
    realistict_control_options

using .forwardmodel: Solution, run_forward_model
using .comparisons: ComparisonPoints, ComparisonPoint, track_segments, TrackSegment
using .visuals
using .Run: run_mtm
using .trial: Trial, trimtrial, get_varvalue_at_frameidx
using .io: PATHS, load_trials, load_cached_trials

end # module
