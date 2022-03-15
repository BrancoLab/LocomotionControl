module jcontrol
# using Term
# install_term_logger()

using Plots
using Interpolations
import Images

include("utils.jl")
include("kinematics.jl")
include("io.jl")
include("track.jl")
include("bike.jl")
include("control.jl")
include("forward_model.jl")

export Track, get_track_borders
export State, Bicycle
export ControlOptions, create_and_solve_control, Bounds
export Solution, run_forward_model
export summary_plot
export PATHS, load_trials
export arena
export DynamicsProblem, KinematicsProblem, realistict_control_options

arena = Images.load("src/arena.png")

using .io: PATHS, load_trials
using .bicycle: State, Bicycle
using .control: ControlOptions,
        create_and_solve_control,
        Bounds,
        State,
        DynamicsProblem,
        KinematicsProblem,
        realistict_control_options

using .forwardmodel: Solution, run_forward_model

include("visuals.jl")

end # module
