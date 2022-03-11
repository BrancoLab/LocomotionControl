module Run


using jcontrol


function run(problemtype::Symbol, num_supports; showtrials::Union{Nothing, Int64}=false)
    problemtype = problemtype == :kinematics ? KinematicsProblem() : DynamicsProblem()
    δt = 0.01 # Δt for forward integration

    # ---------------------------------------------------------------------------- #    
    #                                DEFINE PARAMERS                               #
    # ---------------------------------------------------------------------------- #
    # create track object
    track = Track(; keep_n_waypoints=-1, resolution=0.001)
    @info "track" track

    # create bike
    bike = Bicycle(; l_r=2, l_f=4, width=1.5, m=25, Iz=3) 

    coptions = ControlOptions(;
        # solver optionsx
        verbose=0,
        n_iter=5000,
        num_supports=num_supports,

        # error bounds
        track_safety=1.0,
        ψ_bounds=Bounds(-35, 35, :angle),

        # controls & variables bounds
        u̇_bounds=Bounds(-10, 50),           # cm/s²
        δ̇_bounds=Bounds(-2.5, 2.5),         # rad/s²

        u_bounds=Bounds(5, 100),            # cm
        δ_bounds=Bounds(-80, 80, :angle),   # deg
    )

    # define initial and final conditions
    icond = State(; x=track.X[1], y=track.Y[1], θ=track.θ[1], u=coptions.u_bounds.lower)
    fcond = State(; u=coptions.u_bounds.lower)

    # ---------------------------------------------------------------------------- #
    #                                   FIT MODEL                                  #
    # ---------------------------------------------------------------------------- #
    @info "control options" coptions
    control_model = create_and_solve_control(problemtype, track, bike, coptions, icond, fcond)

    # ---------------------------------------------------------------------------- #
    #                              FORWARD INTEGRATION                             #
    # ---------------------------------------------------------------------------- #
    solution = run_forward_model(problemtype, control_model, icond, bike; δt=δt)

    # ---------------------------------------------------------------------------- #
    #                                 VISUALIZATION                                #
    # ---------------------------------------------------------------------------- #
    # visualize results
    summary_plot(problemtype, control_model, :time)

    # visualize
    trials = !isnothing(showtrials) ? jcontrol.load_trials(; keep_n=showtrials) : nothing
    summary_plot(solution, control_model, track, bike; trials=trials)

    return control_model, solution
end
end




# --------------------------------- Execution -------------------------------- #
# using Term
# install_term_logger()

# print("\n\n" * hLine("start"; style="bold green"))
control_model, solution = @time Run.run(:kinematics, 2000; showtrials=nothing);    

# print(hLine("done"; style="bold blue") * "\n\n")

# TODO forward integration for dyn mod.
# TODO final plots for dyn mod.
