using Plots
import Statistics: median, std
import MyterialColors: black, blue_light
using Term
import Term: track as pbar
install_term_logger()
# install_stacktrace()

using jcontrol
using jcontrol.visuals
import jcontrol: closest_point_idx, euclidean, Track, State
import jcontrol.comparisons: ComparisonPoints

function compare(;  problemtype=:dynamics)
    # ---------------------------------- run MTM --------------------------------- #
    track = Track(;start_waypoint=2, keep_n_waypoints=-1)


    coptions = ControlOptions(;
        u_bounds=Bounds(10, 75),
        δ_bounds=Bounds(-80, 80, :angle),
        δ̇_bounds=Bounds(-4, 4),
        ω_bounds=Bounds(-800, 800, :angle),
        v_bounds=Bounds(-12, 12),
        Fu_bounds=Bounds(-1250, 4500),
    )

    fcond = State(; u=35, n=0, ψ=0)
    final_conditions = fcond

    track, bike, _, solution = run_mtm(
        problemtype,  # model type
        3;  # supports density
        showtrials=nothing,
        control_options=coptions,
        track=track,
        n_iter=5000,
        fcond=final_conditions,
        timed=false,
        showplots=false,
    )

    # plot model trajectory
    plt = draw(:arena)
    draw!(track)
    plot_bike_trajectory!(solution, bike; showbike=false)

    # -------------------------- do comparison with data ------------------------- #
    # load data
    trials = load_cached_trials(; keep_n = 100,)
    cpoints = ComparisonPoints(track; δs=5, trials=trials)

    # show data
    draw!.(trials; lw=3, alpha=.25)    
    draw!.(cpoints.points)

    # do comparison
    speedplot = plot(; title="speed", xlabel="s (cm)", ylabel="speed cm/s", legend=false)
    uplot = plot(; title="u", xlabel="s (cm)", ylabel="u cm/s", legend=false)
    vplot = plot(; title="v", xlabel="s (cm)", ylabel="v cm/s", legend=false)
    ωplot = plot(; title="ang.vel.", xlabel="s (cm)", ylabel="avel rad/s", legend=false)

    U, Ω = Dict{Float64, Vector{Float64}}(), Dict{Float64, Vector{Float64}}()
    for trial in pbar(
        trials; description="Iterating trials", expand=true, columns=:detailed, redirectstdout=false,
    )   
        # plot trial kinematics
        plot!(speedplot, trial.s, trial.speed; color=black, alpha=0.7)
        plot!(uplot, trial.s, trial.u; color=black, alpha=0.7)
        plot!(vplot, trial.s, trial.v; color=black, alpha=0.7)
        plot!(ωplot, trial.s, trial.ω; color=black, alpha=0.7)

        # plot model kinematics at CP
        # for cp in cpoints.points
        #     cp.s < trial.s[1] && continue

        #     if cp.s ∉ keys(U)
        #         U[cp.s] = Vector{Float64}[]
        #         Ω[cp.s] = Vector{Float64}[]
        #     end

        #     # get closest trial point
        #     idx = closest_point_idx(trial.x, cp.x, trial.y, cp.y)
        #     x, y = trial.x[idx], trial.y[idx]

        #     push!(U[cp.s], trial.u[idx])
        #     push!(Ω[cp.s], trial.ω[idx])

        #     # mark point for debugging
        #     scatter!(plt, [x], [y]; ms=5, color="blue", label=nothing, alpha=0.5)
        # end
    end

    # plot avg/std of trials kinematics
    # for cp in cpoints.points
    #     scatter!(uplot, [cp.s], [median(U[cp.s])]; color=blue_light, ms=10)
    #     scatter!(ωplot, [cp.s], [median(Ω[cp.s])]; color=blue_light, ms=10)
    # end

    # plot model kinematics at CP
    for cp in cpoints.points
        # get the closest model point
        idxs = closest_point_idx(solution.x, cp.x, solution.y, cp.y)

        if problemtype==:kinematics
            speed = solution.u[idxs]
            u = solution.u[idxs] * cos(solution.β[idxs])
            v = solution.u[idxs] * sin(solution.β[idxs])
        else
            u, v = solution.u[idxs], solution.v[idxs]
            # speed = u * acos(β)
            speed = sqrt(u^2 + v^2)
            # u = speed * cos(β)
            # v = speed * sin(β)
        end

        scatter!(speedplot, [cp.s], [speed]; color="red", ms=10)
        scatter!(uplot, [cp.s], [u]; color="red", ms=10)
        scatter!(vplot, [cp.s], [-v]; color="red", ms=10)
        scatter!(ωplot, [cp.s], [solution.ω[idxs]]; color="red", ms=10)
    end

    # ------------------------------------ fin ----------------------------------- #

    l = @layout [
        a{0.5w} grid(4, 1)
    ]
    return display(
        plot(
            plt,
            speedplot,
            uplot,
            vplot,
            ωplot;
            layout=l,
            size=(1600, 1200),
        ),
    )
end

compare()

