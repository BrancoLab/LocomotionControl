
using jcontrol
import jcontrol: closest_point_idx, euclidean
import InfiniteOpt: termination_status
import Term.progress: track as pbar
using Term
install_term_logger()

"""
Code to run kinematic model with a range of parameters
and estimate the quality of fit to the tracking data
"""

mse(Δx) = 1 ./ length(Δx) .* sum(Δx .^ 2)

struct ParamEstimationResults
    u̇::Number
    δ̇::Number
    δ::Number
    ℓ::Float64
end

function run_model_fit(params_ranges)
    # create track object
    track = Track()

    # create bike
    bike = Bicycle()

    # initial conditions
    icond = State(; x=track.X[1], y=track.Y[1], u=5)
    fcond = State(; u=5)

    # load data
    trials = load_trials(; keep_n=20)
    cpoints = get_comparison_points(track; δs=10)

    # iterate over parameters values
    _u̇ = params_ranges["u̇_bounds"]
    _δ̇ = params_ranges["δ̇_bounds"]
    _δ = params_ranges["δ_bounds"]
    # _ω = params_ranges["ω_bounds"]
    results::Vector{ParamEstimationResults} = []
    for u̇ in _u̇, δ̇ in _δ̇, δ in _δ
        @info "[bold red]RUNNING[/bold red]" u̇ δ̇ δ
        coptions = ControlOptions(;
            # controls & variables bounds
            u̇_bounds=Bounds(-u̇, u̇),
            δ̇_bounds=Bounds(-δ̇, δ̇),
            u_bounds=Bounds(5, 80),
            δ_bounds=Bounds(-δ, δ, :angle),
            ω_bounds=Bounds(-400, 400, :angle),
        )

        # solve
        control_model = create_and_solve_control(
            KinematicsProblem(), 300, track, bike, coptions, icond, fcond; quiet=true
        )

        if "LOCALLY_SOLVED" != string(termination_status(control_model))
            @warn "Skipping analysis because did not solve locally" termination_status(
                control_model
            )
            continue
        end

        solution = run_forward_model(track, control_model)

        # estimate error
        Δu::Vector{Float64} = []
        Δω::Vector{Float64} = []
        for trial in pbar(
            eachrow(trials); description="Iterating trials", expand=true, columns=:detailed
        )
            trial = Trial(trial, track)

            for cp in cpoints.points
                cp.s < trial.s[1] && continue

                # get closest trial point
                idx = closest_point_idx(trial.x, cp.x, trial.y, cp.y)
                idxs = closest_point_idx(solution.x, cp.x, solution.y, cp.y)

                # get errors
                push!(Δu, solution.u[idxs] - trial.u[idx])
                push!(Δω, solution.ω[idxs] - trial.ω[idx])
            end
        end

        # compute the total error
        ℓ = mse(Δu) + mse(Δω)
        push!(results, ParamEstimationResults(u̇, δ̇, δ, ℓ))
    end

    return results
end

params_ranges = Dict(
    "u̇_bounds" => 100:25:200,
    "δ̇_bounds" => 1:1:7,
    "δ_bounds" => 50:15:120,
    # "ω_bounds" => 500:250:1000,
)

# run
res = run_model_fit(params_ranges)

costs = [sim.ℓ for sim in res]
bestidx = argmin(costs)
print(res[30])
