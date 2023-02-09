cd("/Users/federicoclaudi/Documents/Github/LocomotionControl/analysis/behavior")
using Pkg: Pkg;
Pkg.activate(".")

include(
    "/Users/federicoclaudi/Documents/Github/LocomotionControl/analysis/behavior/multiple_horizons/simulate.jl",
)

"""
Run a simulation in which the model can only plan for `planning_horizon` seconds ahead.
"""
function run_simulation_and_plot(; planning_horizon::Float64=0.5, n_iter=480, Δt=0.01)
    # run global solution
    track = Track(; start_waypoint=2, keep_n_waypoints=-1)

    _, bike, _, globalsolution = run_mtm(
        :dynamics,
        3;
        showtrials=nothing,
        track=track,
        n_iter=5000,
        fcond=State(; u=30, ω=0),
        timed=false,
        showplots=false,
    )

    # plot background & global trajectory
    colors = [
        range(
            HSL(colorant"red"); stop=HSL(colorant"green"), length=(Int ∘ floor)(n_iter / 2)
        )...,
        range(
            HSL(colorant"green"); stop=HSL(colorant"blue"), length=(Int ∘ ceil)(n_iter / 2)
        )...,
    ]

    simtracker = SimTracker(; Δt=Δt)
    p1 = draw(:arena)
    draw!(FULLTRACK; border_alpha=0.25, alpha=0.0)

    for i in pbar(
        1:n_iter;
        redirectstdout=false,
        description="Running horizon: $planning_horizon seconds",
        expand=true,
    )
        simtracker.iter = i

        # run simulation and store results
        initial_state, solution, shouldstop, track = step(
            simtracker, globalsolution, planning_horizon
        )
        shouldstop && break

        # plot stuff

        # draw!(track; color=colors[i], border_lw=5, alpha=0.0)
        i % 5 == 0 && plot_bike_trajectory!(
            solution,
            bike;
            showbike=false,
            label=nothing,
            color=colors[i],
            alpha=0.8,
            lw=4,
        )

        if i % 10 == 0
            draw!(initial_state; color=colors[i], alpha=1)
        end
        # plot!(; title="T: $(round(simtracker.t; digits=2))s | horizon: $planning_horizon s")

        simtracker.s > 260 && break
    end

    return display(p1)
end

run_simulation_and_plot(; planning_horizon=0.75)
println("done")
