module Run

using IOCapture: IOCapture
using TimerOutputs
import Term: RenderableText, Panel
import MyterialColors: green_lighter

using jcontrol
import jcontrol: Track
import ..bicycle: State, Bicycle
import ..control: ControlOptions, realistict_control_options, default_control_options

const to = TimerOutput()

export run_mtm

function run_mtm(
    problemtype::Symbol,
    supports_density::Number;
    track::Union{Nothing,Track}=nothing,
    icond::Union{Nothing,State}=nothing,
    fcond::Union{Nothing,State, Symbol}=nothing,
    control_options::Union{ControlOptions,Symbol}=:default,
    showtrials::Union{Nothing,Int64}=nothing,
    n_iter::Int=1000,
    tol::Float64=1e-12,
    verbose::Int=0,
    timed::Bool=false,
    showplots::Bool=true,
    quiet::Bool=false,
    α::Float64=0.0, # cost of Fu
    γ::Float64=0.0,  # cost of δ̇
    bike::Union{Nothing, Bicycle}=nothing,
    waypoint=nothing
)
    problemtype = problemtype == :kinematics ? KinematicsProblem() : DynamicsProblem()
    δt = 0.01 # Δt for forward integration

    # ---------------------------------------------------------------------------- #    
    #                                DEFINE PARAMERS                               #
    # ---------------------------------------------------------------------------- #
    # create track object
    track = isnothing(track) ? FULLTRACK : track

    # create bike
    bike = isnothing(bike) ? Bicycle() : bike

    # get control options
    if control_options == :default
        control_options = default_control_options
    elseif control_options == :realistic
        control_options = realistict_control_options
    end
    @assert control_options isa ControlOptions "Control options is not a ControlOptions type: $(typeof(control_options)) $control_options"

    # define initial and final conditions
    icond = isnothing(icond) ? State(; x=track.X[1], y=track.Y[1], u=25, ω=2, ψ=.1) : icond
    # fcond = isnothing(fcond) ? State(; u=40, n=0, ψ=0) : fcond

    # ---------------------------------------------------------------------------- #
    #                                   FIT MODEL                                  #
    # ---------------------------------------------------------------------------- #
    # supports_density = 1 -> 100 supports for the whole track, adjust by track length
    n_supports = (Int ∘ round)(supports_density * 100 * (track.S[end] - track.S[1]) / 261)
    # @info "Running with" n_supports
    control_model = @timeit to "solve control" create_and_solve_control(
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
        α=α,
        γ=γ,
        waypoint=waypoint,
    )

    # ---------------------------------------------------------------------------- #
    #                              FORWARD INTEGRATION                             #
    # ---------------------------------------------------------------------------- #
    solution = @timeit to "run forward model" run_forward_model(
        problemtype, track, control_model; δt=δt
    )

    # ---------------------------------------------------------------------------- #
    #                                 VISUALIZATION                                #
    # ---------------------------------------------------------------------------- #
    if showplots
        # visualize results
        @timeit to "plot control" summary_plot(problemtype, control_model, :time)

        # visualize
        trials = if !isnothing(showtrials)
            jcontrol.load_trials(; keep_n=showtrials, method=:efficient)
        else
            nothing
        end
        @timeit to "plot ODEs" summary_plot(
            solution, control_model, track, bike; trials=trials
        )
    end

    # -------------------------------- timing info ------------------------------- #
    # show timing/allocation data
    if timed && !quiet
        c = IOCapture.capture() do
            show(to)
        end
        print(
            "\n\n" / Panel(
                RenderableText(c.output, green_lighter);
                style="green dim",
                title="Timing/Allocations info",
                title_style="green underline bold",
                justify=:center,
            ),
        )
    end

    return track, bike, control_model, solution
end
end
