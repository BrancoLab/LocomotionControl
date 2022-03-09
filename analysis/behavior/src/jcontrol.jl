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

    export Track, get_track_borders
    export State, Bicycle
    export ControlOptions, create_and_solve_control, Bounds
    export Solution, run_forward_model
    export summary_plot

    using .bicycle: State, Bicycle
    using .control: ControlOptions, create_and_solve_control, Bounds, State
    using .forwardmodel: Solution, run_forward_model

    include("visuals.jl")


end # module
