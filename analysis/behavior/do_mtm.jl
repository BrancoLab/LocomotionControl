module Run
# using Term
# install_term_logger()

using jcontrol

# print("\n\n")
# print(hLine("start"; style="bold green"))
#
# print(Panel("""
#             Code to run the Kinematic Bicycle model on the full hairpin maze
#             and solve the minimum time maneuvering problem.
#
#             The MTM is solved in the track's curvilinear coordinates system and
#             then the solution is integrated to get the allocentric coordinates
#             trajectory over time.
#             """))

function run(num_supports; showtrials::Union{Nothing, Int64}=false)

    δt = 0.01 # Δt for forward integration

    # ---------------------------------------------------------------------------- #
    #                                DEFINE PARAMERS                               #
    # ---------------------------------------------------------------------------- #
    # create track object
    track = Track(; width=3, keep_n_waypoints=-1, resolution=0.001)
    @info "track" track

    # create bike
    bike = Bicycle(; L=6, l=2, width=1.5)

    coptions = ControlOptions(;
        # solver optionsx
        verbose=0,
        n_iter=200,
        num_supports=num_supports,

        # error bounds
        track_safety=1.0,
        ψ_bounds=Bounds(-35, 35, :angle),

        # controls & variables bounds
        uv_bounds=Bounds(-50, 50),          # cm/s²
        uδ_bounds=Bounds(-2, 2),        # rad/s²
        v_bounds=Bounds(5, 100),           # cm
        δ_bounds=Bounds(-80, 80, :angle),   # deg
    )

    # define initial and final conditions
    icond = State(; x=track.X[1], y=track.Y[1], θ=track.θ[1], v=coptions.v_bounds.lower)
    fcond = State(; v=coptions.v_bounds.lower)

    # ---------------------------------------------------------------------------- #
    #                                   FIT MODEL                                  #
    # ---------------------------------------------------------------------------- #
    @info "control options" coptions
    control_model = create_and_solve_control(track, bike, coptions, icond, fcond)

    # ---------------------------------------------------------------------------- #
    #                              FORWARD INTEGRATION                             #
    # ---------------------------------------------------------------------------- #
    solution = run_forward_model(control_model, icond, bike; δt=δt)

    # ---------------------------------------------------------------------------- #
    #                                 VISUALIZATION                                #
    # ---------------------------------------------------------------------------- #
    # visualize results
    summary_plot(control_model, :time)

    # visualize
    trials = !isnothing(showtrials) ? jcontrol.load_trials(; keep_n=showtrials) : nothing
    summary_plot(solution, control_model, track, bike; trials=trials)

    # print("\n")
    # print(hLine("done"; style="bold blue"))
    # print("\n\n")
    #
    return nothing
end
end




# --------------------------------- Execution -------------------------------- #
Run.run(2000; showtrials=50)


