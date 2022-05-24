using Plots
import Statistics: median, std
import MyterialColors: black, blue_light
using Term
using Term.progress
import InfiniteOpt: objective_value
install_term_logger()
# install_stacktrace()

using jcontrol
using jcontrol.visuals
import jcontrol: closest_point_idx, Track, State
import jcontrol.comparisons: ComparisonPoints
import jcontrol.bicycle: Bicycle



function compare(;  problemtype=:dynamics)
    # ---------------------------------- run MTM --------------------------------- #
    track = Track(; start_waypoint=4, keep_n_waypoints=-1)


    coptions = ControlOptions(;
    u_bounds=Bounds(10, 80),
    δ_bounds=Bounds(-50, 50, :angle),
    δ̇_bounds=Bounds(-6, 6),
    ω_bounds=Bounds(-400, 400, :angle),
    v_bounds=Bounds(-15, 15),
    Fu_bounds=Bounds(-2000, 3000),
    )

    bike = Bicycle(;
        l_f=3, l_r=2, c=4000, width=2.0
    )

    track, bike, _, solution = run_mtm(
        problemtype,  # model type
        2;  # supports density
        showtrials=nothing,
        control_options=coptions,
        bike=bike,
        track=track,
        n_iter=5000,
        fcond=State(; u=30, ψ=0, n=-.5),
        timed=false,
        showplots=false,
    )
    @info "Duration" solution.t[end]

    # plot model trajectory
    plt = draw(:arena)
    draw!(track)


    # -------------------------- do comparison with data ------------------------- #
    # load data
    trials = load_cached_trials(; keep_n = 100,)
    cpoints = ComparisonPoints(track; δs=5, trials=trials)
    fasttrials = filter(t -> t.duration <= solution.t[end], trials)

    # print percentage of fast trials (roudned)
    @info "Fast trials: $(round(100 * length(fasttrials) / length(trials)))% | ($(length(fasttrials))/$(length(trials)))"


    # show data
    draw!.(trials; lw=2, alpha=.15)    
    draw!.(cpoints.points)
    plot_bike_trajectory!(solution, bike; showbike=false)

    scatter!(
        solution.x, solution.y, marker_z=solution.ω, c=:bwr, label=nothing
    )
    # do comparison
    speedplot = plot(; title="speed", xlabel="s (cm)", ylabel="speed cm/s", legend=false)
    uplot = plot(; title="u", xlabel="s (cm)", ylabel="u cm/s", legend=false)
    vplot = plot(; title="v", xlabel="s (cm)", ylabel="v cm/s", legend=false)
    ωplot = plot(; title="ang.vel.", xlabel="s (cm)", ylabel="avel rad/s", legend=false)

    @track for trial in trials[1:100]
        # plot trial kinematics
        plot!(speedplot, trial.s, trial.speed; color=black, alpha=0.7)
        plot!(uplot, trial.s, trial.u; color=black, alpha=0.7)
        plot!(vplot, trial.s, trial.v; color=black, alpha=0.7)
        plot!(ωplot, trial.s, trial.ω; color=black, alpha=0.7)
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
            speed = sqrt(u^2 + v^2)
        end

        scatter!(speedplot, [cp.s], [speed]; color="red", ms=10)
        scatter!(uplot, [cp.s], [u]; color="red", ms=10)
        scatter!(vplot, [cp.s], [v]; color="red", ms=10)
        scatter!(ωplot, [cp.s], [solution.ω[idxs]]; color="red", ms=10)
    end


    # ------------------------------------ fin ----------------------------------- #
    h = histogram(
        map(t->t.duration, trials), color="black", label=nothing, xlim=[0, 15]
    )
    duration = solution.t[end]

    plot!(h, [duration, duration], [0, 200], lw=5, alpha=.8, color="red", label=nothing)


    l = @layout [
        a{0.5w} grid(5, 1)
    ]
    return display(
        plot(
            plt,
            speedplot,
            uplot,
            vplot,
            ωplot,
            h;
            layout=l,
            size=(1600, 1200),
        ),
    )
end


print("\n\n" * hLine("start"; style="bold green"))

compare()

print("\n\n" * hLine("finish"; style="bold blue"))
