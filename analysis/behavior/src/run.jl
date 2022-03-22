module Run

import IOCapture
using TimerOutputs
import Term: RenderableText, Panel
import MyterialColors: green_lighter

using jcontrol
import jcontrol: Track
import ..bicycle: State
import ..control: ControlOptions, realistict_control_options, default_control_options

const to = TimerOutput()

export run_mtm

function run_mtm(
        problemtype::Symbol,
        supports_density::Number;
        track::Union{Nothing, Track}=nothing,
        icond::Union{Nothing, State}=nothing,
        fcond::Union{Nothing, State}=nothing,
        control_options::Union{ControlOptions, Symbol} = :default,
        showtrials::Union{Nothing, Int64}=nothing,
        n_iter::Int=1000,
        tol::Float64=1e-12,
        verbose::Int=0,
        timed::Bool=false,
        showplots::Bool=true,
        quiet::Bool=false,
    )

    problemtype = problemtype == :kinematics ? KinematicsProblem() : DynamicsProblem()
    δt = 0.01 # Δt for forward integration

    # ---------------------------------------------------------------------------- #    
    #                                DEFINE PARAMERS                               #
    # ---------------------------------------------------------------------------- #
    # create track object
    track = isnothing(track) ? Track(; start_waypoint=2, keep_n_waypoints=-1) : track

    # create bike
    bike = Bicycle() 

    # get control options
    if control_options == :default
        control_options = default_control_options
    elseif control_options == :realistic
        control_options == realistict_control_options
    end

    # define initial and final conditions
    icond = isnothing(icond) ? State(; x=track.X[1], y=track.Y[1], u=10) : icond
    fcond = isnothing(fcond) ? State(; u=15, n=0, ψ=0) : fcond

    # ---------------------------------------------------------------------------- #
    #                                   FIT MODEL                                  #
    # ---------------------------------------------------------------------------- #
    # supports_density = 1 -> 100 supports for the whole track, adjust by track length
    n_supports = (Int ∘ round)(supports_density * 100 * track.S_f/261)
    control_model = @timeit to "solve control"  create_and_solve_control(
                        problemtype,
                        n_supports,
                        track,
                        bike,
                        control_options,
                        icond,
                        fcond; 
                        n_iter=n_iter,
                        tollerance=tol,
                        verbose=verbose,
                        quiet=quiet,
    )               
 
    # ---------------------------------------------------------------------------- #
    #                              FORWARD INTEGRATION                             #
    # ---------------------------------------------------------------------------- #
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
    if timed && !quiet
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
