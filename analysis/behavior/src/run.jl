module Run

import IOCapture
using TimerOutputs
import Term: RenderableText, Panel
import MyterialColors: green_lighter

using jcontrol

const to = TimerOutput()

export run_mtm

function run_mtm(
        problemtype::Symbol,
        num_supports; 
        realistic_controls::Bool = false,
        showtrials::Union{Nothing, Int64}=nothing,
        niters::Int=250,
        verbose::Int=0,
        timed::Bool=false,
        showplots::Bool=true,
    )

    problemtype = problemtype == :kinematics ? KinematicsProblem() : DynamicsProblem()
    δt = 0.01 # Δt for forward integration

    # ---------------------------------------------------------------------------- #    
    #                                DEFINE PARAMERS                               #
    # ---------------------------------------------------------------------------- #
    # create track object
    track = Track(; keep_n_waypoints=-1, resolution=0.00001)


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
            ω_bounds=realistict_control_options["ω"]
        )
    else
        coptions = ControlOptions(;
            # solver optionsx
            verbose=verbose,
            n_iter=niters,
            num_supports=num_supports,

            # error bounds
            track_safety=1.0,
            ψ_bounds=Bounds(-35, 35, :angle),

            # controls & variables bounds
            u_bounds=Bounds(5, 80),            # cm
            u̇_bounds=Bounds(-180, 200),         # cm/s²
            δ_bounds=Bounds(-80, 80, :angle),  # deg
            δ̇_bounds=Bounds(-4, 4),            # rad/s²
            ω_bounds=Bounds(-500, 500, :angle)
        )
    end

    @info "using" problemtype realistic_controls coptions.u_bounds coptions.u̇_bounds coptions.δ_bounds coptions.δ̇_bounds  coptions.ω_bounds

    # define initial and final conditions
    icond = State(; x=track.X[1], y=track.Y[1], ψ=0, u=25)
    fcond = State(; u=5)
    # icond = State(; x=track.X[1], y=track.Y[1], θ=track.θ[1], u=25, ω=8)
    # fcond = State(; u=coptions.u_bounds.lower, ω=0)

    # ---------------------------------------------------------------------------- #
    #                                   FIT MODEL                                  #
    # ---------------------------------------------------------------------------- #
    control_model = @timeit to "solve control"  create_and_solve_control(problemtype, track, bike, coptions, icond, fcond)
 
    # ---------------------------------------------------------------------------- #
    #                              FORWARD INTEGRATION                             #
    # ---------------------------------------------------------------------------- #
    # solution = @timeit to "run forward model"  run_forward_model(problemtype, control_model, icond, bike; δt=δt)
    solution = @timeit to "run forward model"  run_forward_model(track, control_model; δt=δt)

    # ---------------------------------------------------------------------------- #
    #                                 VISUALIZATION                                #
    # ---------------------------------------------------------------------------- #
    if showplots
        # visualize results
        @timeit to "plot control" summary_plot(problemtype, control_model, :time)

        # visualize
        trials = !isnothing(showtrials) ? jcontrol.load_trials(; keep_n=showtrials, method=:efficient) : nothing
        @timeit to "plot ODEs" summary_plot(solution, control_model, track, bike; trials=trials)
    end

    # -------------------------------- timing info ------------------------------- #
    # show timing/allocation data
    if timed
        c = IOCapture.capture() do
            show(to)
        end
        print(
            "\n\n" / Panel(
                RenderableText(c.output, green_lighter),
                style="green dim",
                title="Timing/Allocations info",
                title_style="green underline bold",
                justify=:center
            )
        )
    end

    return track, bike, control_model, solution
end
end
