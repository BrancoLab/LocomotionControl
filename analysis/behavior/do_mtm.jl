module Run


using jcontrol
using TimerOutputs

const to = TimerOutput()


function run(
        problemtype::Symbol,
        num_supports; 
        realistic_controls::Bool = false,
        showtrials::Union{Nothing, Int64}=false,
        niters::Int=250,
        verbose::Int=0,
    )

    problemtype = problemtype == :kinematics ? KinematicsProblem() : DynamicsProblem()
    δt = 0.01 # Δt for forward integration

    # ---------------------------------------------------------------------------- #    
    #                                DEFINE PARAMERS                               #
    # ---------------------------------------------------------------------------- #
    # create track object
    track = Track(; keep_n_waypoints=-1, resolution=0.001)
    @info "track" track

    # create bike
    bike = Bicycle() 

    if realistic_controls
        coptions = ControlOptions(
            # solver optionsx
            verbose=verbose,
            n_iter=niters,
            num_supports=num_supports,

            # error bounds
            track_safety=1.0,
            ψ_bounds=Bounds(-45, 45, :angle),

            # controls & variables bounds
            u̇_bounds=realistict_control_options["u̇"],   # cm/s²
            δ̇_bounds=realistict_control_options["δ̇"],   # rad/s²
            u_bounds=realistict_control_options["u"],   # cm
            δ_bounds=realistict_control_options["δ"],   # deg
        )
    else
        coptions = ControlOptions(;
            # solver optionsx
            verbose=verbose,
            n_iter=niters,
            num_supports=num_supports,

            # error bounds
            track_safety=1.0,
            ψ_bounds=Bounds(-45, 45, :angle),

            # controls & variables bounds
            u_bounds=Bounds(5, 80),            # cm
            u̇_bounds=Bounds(-60, 120),         # cm/s²
            δ_bounds=Bounds(-90, 90, :angle),  # deg
            δ̇_bounds=Bounds(-6, 6),            # rad/s²
        )
    end

    # define initial and final conditions
    icond = State(; x=track.X[1], y=track.Y[1], θ=track.θ[1], u=coptions.u_bounds.lower)
    fcond = State(; u=coptions.u_bounds.lower)

    # ---------------------------------------------------------------------------- #
    #                                   FIT MODEL                                  #
    # ---------------------------------------------------------------------------- #
    control_model = @timeit to "solve control"  create_and_solve_control(problemtype, track, bike, coptions, icond, fcond)

    # ---------------------------------------------------------------------------- #
    #                              FORWARD INTEGRATION                             #
    # ---------------------------------------------------------------------------- #
    solution = @timeit to "run forward model"  run_forward_model(problemtype, control_model, icond, bike; δt=δt)

    # ---------------------------------------------------------------------------- #
    #                                 VISUALIZATION                                #
    # ---------------------------------------------------------------------------- #
    # visualize results
    @timeit to "plot control" summary_plot(problemtype, control_model, :time)

    # visualize
    trials = !isnothing(showtrials) ? jcontrol.load_trials(; keep_n=showtrials) : nothing
    @timeit to "plot ODEs" summary_plot(solution, control_model, track, bike; trials=trials)

    show(to)

    return control_model, solution
end
end




# --------------------------------- Execution -------------------------------- #
# using Term
# install_term_logger()

# print("\n\n" * hLine("start"; style="bold green"))
control_model, solution = Run.run(
    :kinematics,
    200;
    realistic_controls=false,
    showtrials=50,
    niters=1000
)

# print(hLine("done"; style="bold blue") * "\n\n")

# TODO forward integration for dyn mod.
# TODO final plots for dyn mod.

nothing
