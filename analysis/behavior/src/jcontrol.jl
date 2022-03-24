module jcontrol
using Term
install_term_logger()

install_stacktrace

using Plots
using Interpolations

include("utils.jl")
include("kinematics.jl")
include("io.jl")
include("track.jl")
include("bike.jl")
include("control.jl")
include("forward_model.jl")
include("comparisons.jl")
include("visuals.jl")
include("run.jl")
include("trial.jl")

export Track, get_track_borders
export State, Bicycle
export ControlOptions, create_and_solve_control, Bounds
export Solution, run_forward_model
export summary_plot
export PATHS, load_trials
export arena
export DynamicsProblem, KinematicsProblem, realistict_control_options
export get_comparison_points
export run_mtm
export Trial

using .io: PATHS, load_trials
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
using .comparisons: ComparisonPoints, ComparisonPoint, get_comparison_points
using .visuals:
    plot_arena,
    plot_arena!,
    plot_track!,
    summary_plot,
    plot_trials!,
    plot_comparison_point!,
    plot_bike_trajectory!
using .Run: run_mtm
using .trial: Trial

end # module
