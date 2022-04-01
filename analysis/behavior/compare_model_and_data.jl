using Plots
import MyterialColors: black
using Term
import Term: track as pbar
install_term_logger()
# install_stacktrace()

using jcontrol
using jcontrol.visuals
import jcontrol: closest_point_idx, euclidean, Track
import jcontrol.comparisons: ComparisonPoints

function compare(;  problemtype=:dynamics)
    # ---------------------------------- run MTM --------------------------------- #
    track = Track(;start_waypoint=2, keep_n_waypoints=-1)


    coptions = ControlOptions(;
        u_bounds=Bounds(10, 80),
        δ_bounds=Bounds(-150, 150, :angle),
        δ̇_bounds=Bounds(-5, 5),
        ω_bounds=Bounds(-800, 800, :angle),
        v_bounds=Bounds(-1000, 1000),
        Fu_bounds=Bounds(-3000, 3000),
    )

    track, bike, _, solution = run_mtm(
        problemtype,  # model type
        3;  # supports density
        showtrials=nothing,
        control_options=coptions,
        track=track,
        n_iter=5000,
        timed=false,
        showplots=false,
    )

    # plot model trajectory
    plt = draw(:arena)
    draw!(track)
    plot_bike_trajectory!(solution, bike; showbike=false)

    # -------------------------- do comparison with data ------------------------- #
    # load data
    trials = load_cached_trials(; keep_n = 20,)
    cpoints = ComparisonPoints(track; δs=10, trials=trials)

    # show data
    draw!.(trials; lw=3)    
    draw!.(cpoints.points)

    # do comparison
    speedplot = plot(; title="speed", xlabel="s (cm)", ylabel="speed cm/s", legend=false)
    uplot = plot(; title="u", xlabel="s (cm)", ylabel="u cm/s", legend=false)
    vplot = plot(; title="v", xlabel="s (cm)", ylabel="v cm/s", legend=false)
    ωplot = plot(; title="ang.vel.", xlabel="s (cm)", ylabel="avel rad/s", legend=false)

    for trial in pbar(
        trials; description="Iterating trials", expand=true, columns=:detailed, redirectstdout=false,
    )   
        # plot trial kinematics
        plot!(speedplot, trial.s, trial.speed; color=black, alpha=0.7)
        plot!(uplot, trial.s, trial.u; color=black, alpha=0.7)
        plot!(vplot, trial.s, trial.v; color=black, alpha=0.7)
        plot!(ωplot, trial.s, trial.ω; color=black, alpha=0.7)

        # plot model kinematics at CP
        for cp in cpoints.points
            cp.s < trial.s[1] && continue

            # get closest trial point
            idx = closest_point_idx(trial.x, cp.x, trial.y, cp.y)
            x, y, θ = trial.x[idx], trial.y[idx], trial.θ[idx]
            # mark point for debugging
            scatter!(plt, [x], [y]; ms=5, color="blue", label=nothing, alpha=0.5)
        end
    end


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
        scatter!(vplot, [cp.s], [v]; color="red", ms=10)
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

