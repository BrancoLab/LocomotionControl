module jcontrol
# using Term
# install_term_logger()

using Plots
using Interpolations

include("utils.jl")
include("track.jl")
include("bike.jl")
include("control.jl")
include("forward_model.jl")
include("run.jl")
include("visuals.jl")

export Track, get_track_borders, FULLTRACK
export State, Bicycle
export ControlOptions, create_and_solve_control, Bounds
export Solution, run_forward_model
export summary_plot
export arena
export DynamicsProblem, KinematicsProblem, realistict_control_options
export run_mtm

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
using .visuals
using .Run: run_mtm

end # module
