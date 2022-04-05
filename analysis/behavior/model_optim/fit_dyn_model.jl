import Statistics: mean
import InfiniteOpt: termination_status, objective_value
import Term.progress: ProgressBar, update, start, stop
using Term
install_term_logger()

using jcontrol
import jcontrol: closest_point_idx, toDict
import jcontrol.comparisons: σ, ComparisonPoints


"""
Code to run kinematic model with a range of parameters
and estimate the quality of fit to the tracking data
"""

struct ParamEstimationResults
    δ̇::Number
    δ::Number
    ω::Number
    v::Number
    Fu::Number
    duration::Number
    ℓ::Float64
end



function run_model_fit(params_ranges)
    # create track object
    track = Track()

    # create bike
    bike = Bicycle()

    # initial conditions
    icond = State(; u=10)
    fcond = State(; u=5)

    # load data
    trials = load_cached_trials(; keep_n = 20,)
    cpoints = ComparisonPoints(track; δs=10, trials=trials)

    # iterate over parameters values
    _δ̇ = params_ranges["δ̇_bounds"]
    _δ = params_ranges["δ_bounds"]
    _ω = params_ranges["ω_bounds"]
    _v = params_ranges["v_bounds"]
    _Fu = params_ranges["Fu_bounds"]
    results::Vector{ParamEstimationResults} = []

    nsims = *(length.(values(params_ranges))...)
    pbar = ProgressBar(; N=nsims, redirectstdout=false)
    start(pbar)
    for δ̇ in _δ̇, δ in _δ, ω in _ω, v in _v, Fu in _Fu
        update(pbar)

        coptions = ControlOptions(
            # controls & variables bounds
            u_bounds=Bounds(5, 80),
            δ̇_bounds=Bounds(-δ̇, δ̇),
            δ_bounds=Bounds(-δ, δ, :angle),
            ω_bounds=Bounds(-ω, ω, :angle),
            v_bounds = Bounds(-v, v),
            Fu_bounds = Bounds(-Fu, Fu),
        )
        
        # solve
        control_model = create_and_solve_control(
                DynamicsProblem(),            
                300,
                track,
                bike,
                coptions,
                icond,
                fcond;
                n_iter=5000,
                quiet=true)

        if "LOCALLY_SOLVED" != string(termination_status(control_model))
            @warn "Skipping analysis because did not solve locally" termination_status(control_model)
            continue
        end

        solution = run_forward_model(DynamicsProblem(), track, control_model)

        # estimate error
        σx::Vector{Float64} = []
        σy::Vector{Float64} = []
        σθ::Vector{Float64} = []
        σu::Vector{Float64} = []
        σω::Vector{Float64} = []
        σv::Vector{Float64} = []

        for cp in cpoints.points
            # get closest trial point
            idx = closest_point_idx(solution.x, cp.x, solution.y, cp.y)

            # get errors
            push!(σx, σ(solution.x[idx], cp.kinematics.x; use=:med))
            push!(σy, σ(solution.y[idx], cp.kinematics.y; use=:med))
            push!(σθ, σ(solution.θ[idx], cp.kinematics.θ; use=:med))
            push!(σu, σ(solution.u[idx], cp.kinematics.u; use=:med))
            push!(σω, σ(solution.ω[idx], cp.kinematics.ω; use=:med))
            push!(σv, σ(solution.v[idx], cp.kinematics.v; use=:med))
        end

        # get duration
        duration = objective_value(control_model)

        # compute the total error
        ℓ = mean(abs.(σx)) + mean(abs.(σy)) + mean(abs.(σθ)) + mean(abs.(σu)) + mean(abs.(σω))
        # ℓ = mean(abs.(σu)) + mean(abs.(σω)) + mean(abs.(σv))

        push!(
            results, ParamEstimationResults(δ̇, δ, ω, v, Fu, duration, ℓ)
        )
    end
    stop(pbar)
    return results
end




params_ranges = Dict(
    "δ̇_bounds"  => [2, 4, 8],
    "δ_bounds"  => [60, 90, 120],
    "ω_bounds"  => [250, 500, 1000],
    "v_bounds"  => [25, 250, 500],
    "Fu_bounds" => [2000, 3500, 5000],
)

tot = *(length.(values(params_ranges))...)
@info "Number of simulations $tot"

# run
res = run_model_fit(params_ranges)

costs = [sim.ℓ for sim in res]
durations = [sim.duration for sim in res]

bestidx = argmin(costs)
println("The least cost params combinations is", bestidx)
toDict(res[bestidx])