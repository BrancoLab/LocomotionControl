
using jcontrol
import jcontrol: closest_point_idx, euclidean
import InfiniteOpt: termination_status
import Term.progress: ProgressBar, update, start, stop
using Term
install_term_logger()


"""
Code to run kinematic model with a range of parameters
and estimate the quality of fit to the tracking data
"""

mse(Δx) = 1 ./ length(Δx) .* sum(Δx .^ 2)

struct ParamEstimationResults
    δ̇::Number
    δ::Number
    ω::Number
    Fy::Number
    v::Number
    Fu::Number
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
    trials = load_trials(; keep_n = 20,)
    cpoints = get_comparison_points(track; δs=10)

    # iterate over parameters values
    _δ̇ = params_ranges["δ̇_bounds"]
    _δ = params_ranges["δ_bounds"]
    _ω = params_ranges["ω_bounds"]
    _Fy = params_ranges["Fy_bounds"]
    _v = params_ranges["v_bounds"]
    _Fu = params_ranges["Fu_bounds"]
    results::Vector{ParamEstimationResults} = []

    nsims = *(length.(values(params_ranges))...)
    pbar = ProgressBar(; N=nsims, redirectstdout=false)
    start(pbar)
    for δ̇ in _δ̇, δ in _δ, ω in _ω, Fy in _Fy, v in _v, Fu in _Fu
        update(pbar)

        coptions = ControlOptions(
            # controls & variables bounds
            u_bounds=Bounds(5, 80),
            δ̇_bounds=Bounds(-δ̇, δ̇),
            δ_bounds=Bounds(-δ, δ, :angle),
            ω_bounds=Bounds(-ω, ω, :angle),
            Fy_bounds = Bounds(-Fy, Fy),
            v_bounds = Bounds(-v, v),
            Fu_bounds = Bounds(-Fu, Fu),
        )
        
        # solve
        control_model = create_and_solve_control(
                DynamicsProblem(),            
                100,
                track,
                bike,
                coptions,
                icond,
                fcond;
                n_iter=5000,
                quiet=true)

        if "LOCALLY_SOLVED" != string(termination_status(control_model))
            # @warn "Skipping analysis because did not solve locally" termination_status(control_model)
            continue
        end

        solution = run_forward_model(DynamicsProblem(), track, control_model)

        # estimate error
        Δu::Vector{Float64} = []
        Δω::Vector{Float64} = []
        for trial in eachrow(trials)
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
        push!(
            results, ParamEstimationResults(δ̇, δ, ω, Fy, v, Fu, ℓ)
        )
    end
    stop(pbar)
    return results
end


params_ranges = Dict(
    "δ̇_bounds"  => [1, 3, 6],
    "δ_bounds"  => [45, 90],
    "ω_bounds"  => [100, 500, 1000],
    "Fy_bounds" => [250, 500, 1500],
    "v_bounds"  => [50, 250, 1500],
    "Fu_bounds" => [50, 250, 500],
)




tot = *(length.(values(params_ranges))...)
@info "Number of simulations $tot"

run
res = run_model_fit(params_ranges)

costs = [sim.ℓ for sim in res]
bestidx = argmin(costs)
print(res[30])